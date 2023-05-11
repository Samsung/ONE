/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ReduceLayer.h"

#include "OperationUtils.h"

#include "cker/neon/neon_check.h"
#include <cker/operation/Reduce.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
{

namespace
{

template <typename T>
void evalLogic(const IPortableTensor *input, IPortableTensor *output, const std::vector<int> &axes,
               bool keep_dims, T init_value, nnfw::cker::Reduce &reduce_kernel,
               T reducer(const T current, const T in))
{
  reduce_kernel.prepare(input->getShape().rank(), axes.size());
  bool result =
    reduce_kernel.ReduceGeneric<T>(getShape(input), getBuffer<T>(input), getShape(output),
                                   getBuffer<T>(output), axes, keep_dims, init_value, reducer);

  if (!result)
  {
    throw std::runtime_error{"Reduce: Fail to run"};
  }
}

template <typename T>
std::function<void(const IPortableTensor *, IPortableTensor *, const std::vector<int> &)>
evalType(bool keep_dims, nnfw::cker::Reduce &reduce_kernel, ReduceType reduce_type)
{
  switch (reduce_type)
  {
    case ReduceType::kSum:
      return std::bind(&evalLogic<T>, std::placeholders::_1, std::placeholders::_2,
                       std::placeholders::_3, keep_dims, static_cast<T>(0), reduce_kernel,
                       [](const T current, const T in) -> T { return in + current; });
      break;
    case ReduceType::kProd:
      return std::bind(&evalLogic<T>, std::placeholders::_1, std::placeholders::_2,
                       std::placeholders::_3, keep_dims, static_cast<T>(1), reduce_kernel,
                       [](const T current, const T in) -> T { return in * current; });
      break;
    case ReduceType::kMax:
      return std::bind(
        &evalLogic<T>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        keep_dims, std::numeric_limits<T>::lowest(), reduce_kernel,
        [](const T current, const T in) -> T { return (in > current) ? in : current; });
      break;
    case ReduceType::kMin:
      return std::bind(
        &evalLogic<T>, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
        keep_dims, std::numeric_limits<T>::max(), reduce_kernel,
        [](const T current, const T in) -> T { return (in < current) ? in : current; });
      break;
    default:
      throw std::runtime_error{"Reduce: Unsupported reduce type"};
  }
}

// Template specialization for bool type
template <>
std::function<void(const IPortableTensor *, IPortableTensor *, const std::vector<int> &)>
evalType<bool>(bool keep_dims, nnfw::cker::Reduce &reduce_kernel, ReduceType reduce_type)
{
  switch (reduce_type)
  {
    case ReduceType::kAny:
      return std::bind(&evalLogic<bool>, std::placeholders::_1, std::placeholders::_2,
                       std::placeholders::_3, keep_dims, false, reduce_kernel,
                       [](const bool current, const bool in) -> bool { return in || current; });
      break;
    case ReduceType::kAll:
      return std::bind(&evalLogic<bool>, std::placeholders::_1, std::placeholders::_2,
                       std::placeholders::_3, keep_dims, true, reduce_kernel,
                       [](const bool current, const bool in) -> bool { return in && current; });
      break;
    default:
      throw std::runtime_error{"Reduce: Unsupported reduce type"};
  }
}

std::function<void(const IPortableTensor *, IPortableTensor *, const std::vector<int> &)>
generateKernelGeneric(const IPortableTensor *input, bool keep_dims,
                      nnfw::cker::Reduce &reduce_kernel, ReduceType reduce_type)
{
  switch (input->data_type())
  {
    case OperandType::FLOAT32:
      return evalType<float>(keep_dims, reduce_kernel, reduce_type);
    case OperandType::INT32:
      return evalType<int32_t>(keep_dims, reduce_kernel, reduce_type);
    case OperandType::BOOL8:
      return evalType<bool>(keep_dims, reduce_kernel, reduce_type);
    default:
      throw std::runtime_error{"Reduce(generic): unsupported data type"};
  }
}

// TODO Refine this function
void evalSumQuantized(const IPortableTensor *input, IPortableTensor *output,
                      const std::vector<int> &axes, bool keep_dims,
                      nnfw::cker::Reduce &reduce_kernel)
{
  const bool same_scale = (input->data_scale() == output->data_scale() &&
                           input->data_zero_point() == output->data_zero_point());

  reduce_kernel.prepare(input->getShape().rank(), axes.size());

  if (!same_scale)
  {
    std::vector<int32_t> temp_sum(output->getShape().num_elements());
    bool result = reduce_kernel.QuantizedMeanOrSum<uint8_t, int32_t>(
      getBuffer<uint8_t>(input), input->data_zero_point(), input->data_scale(), getShape(input),
      getBuffer<uint8_t>(output), output->data_zero_point(), output->data_scale(), getShape(output),
      axes, keep_dims, temp_sum.data(), true,
      [](const int32_t current, const uint8_t in) -> int32_t {
        const int32_t actual_in = static_cast<int32_t>(in);
        return current + actual_in;
      });

    if (!result)
    {
      throw std::runtime_error{"Reduce: Fail to run"};
    }

    return;
  }

  const auto kernel = generateKernelGeneric(input, keep_dims, reduce_kernel, ReduceType::kSum);
  kernel(input, output, axes);
}

} // namespace

ReduceLayer::ReduceLayer()
  : _input(nullptr), _axes(nullptr), _output(nullptr), _reduce_kernel(new nnfw::cker::Reduce()),
    _kernel(), _reduceType(ReduceType::kInvalid)
{
  // DO NOTHING
}

ReduceLayer::~ReduceLayer() = default;

void ReduceLayer::configure(const IPortableTensor *input, const IPortableTensor *axes,
                            IPortableTensor *output, ReduceType reduceType, bool keep_dims)
{
  _input = input;
  _axes = axes;
  _output = output;
  _reduceType = reduceType;

  switch (_reduceType)
  {
    case ReduceType::kSum:
      if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
      {
        _kernel = std::bind(&evalSumQuantized, std::placeholders::_1, std::placeholders::_2,
                            std::placeholders::_3, keep_dims, *_reduce_kernel);
        return;
      }
      _kernel = generateKernelGeneric(_input, keep_dims, *_reduce_kernel, ReduceType::kSum);
      break;
    case ReduceType::kProd:
      _kernel = generateKernelGeneric(_input, keep_dims, *_reduce_kernel, ReduceType::kProd);
      break;
    case ReduceType::kMax:
      _kernel = generateKernelGeneric(_input, keep_dims, *_reduce_kernel, ReduceType::kMax);
      break;
    case ReduceType::kMin:
      _kernel = generateKernelGeneric(_input, keep_dims, *_reduce_kernel, ReduceType::kMin);
      break;
    case ReduceType::kAny:
      _kernel = generateKernelGeneric(_input, keep_dims, *_reduce_kernel, ReduceType::kAny);
      break;
    case ReduceType::kAll:
      _kernel = generateKernelGeneric(_input, keep_dims, *_reduce_kernel, ReduceType::kAll);
      break;
    default:
      throw std::runtime_error{"Reduce: Unsupported reduce type"};
  }
}

void ReduceLayer::run()
{
  const auto axes = getReducerAxes(_axes);
#ifdef USE_NEON
  int32_t rank = _input->getShape().rank();
  if (_input->data_type() == ir::DataType::FLOAT32 && _reduceType == ReduceType::kSum &&
      axes.size() == 1 && (axes[0] == -1 || axes[0] == rank - 1))
  {
    OptimizedReduceSum(getBuffer<float>(_input), getShape(_input), getBuffer<float>(_output));
    return;
  }
#endif // NEON
  _kernel(_input, _output, axes);
}

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
