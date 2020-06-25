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

#include <cker/operation/Reduce.h>

namespace onert
{
namespace backend
{
namespace cpu
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
  reduce_kernel.prepare(input->num_dimensions(), axes.size());
  bool result = reduce_kernel.ReduceGeneric<T>(
      getTensorShape(input), reinterpret_cast<const T *>(input->buffer()), getTensorShape(output),
      reinterpret_cast<T *>(output->buffer()), axes, keep_dims, init_value, reducer);

  if (!result)
  {
    throw std::runtime_error{"Reduce: Fail to run"};
  }
}

template <typename T>
void evalType(const IPortableTensor *input, IPortableTensor *output, const std::vector<int> &axes,
              bool keep_dims, nnfw::cker::Reduce &reduce_kernel, ReduceType reduce_type)
{
  switch (reduce_type)
  {
    case ReduceType::kSum:
      return evalLogic<T>(input, output, axes, keep_dims, static_cast<T>(0), reduce_kernel,
                          [](const T current, const T in) -> T { return in + current; });
      break;
    case ReduceType::kProd:
      return evalLogic<T>(input, output, axes, keep_dims, static_cast<T>(1), reduce_kernel,
                          [](const T current, const T in) -> T { return in * current; });
      break;
    case ReduceType::kMax:
      return evalLogic<T>(
          input, output, axes, keep_dims, std::numeric_limits<T>::lowest(), reduce_kernel,
          [](const T current, const T in) -> T { return (in > current) ? in : current; });
      break;
    case ReduceType::kMin:
      return evalLogic<T>(
          input, output, axes, keep_dims, std::numeric_limits<T>::max(), reduce_kernel,
          [](const T current, const T in) -> T { return (in < current) ? in : current; });
      break;
    default:
      throw std::runtime_error{"Reduce: Unsupported reduce type"};
  }
}

// Template specialization for bool type
template <>
void evalType<bool>(const IPortableTensor *input, IPortableTensor *output,
                    const std::vector<int> &axes, bool keep_dims, nnfw::cker::Reduce &reduce_kernel,
                    ReduceType reduce_type)
{
  switch (reduce_type)
  {
    case ReduceType::kAny:
      return evalLogic<bool>(
          input, output, axes, keep_dims, false, reduce_kernel,
          [](const bool current, const bool in) -> bool { return in || current; });
      break;
    case ReduceType::kAll:
      return evalLogic<bool>(
          input, output, axes, keep_dims, true, reduce_kernel,
          [](const bool current, const bool in) -> bool { return in && current; });
      break;
    default:
      throw std::runtime_error{"Reduce: Unsupported reduce type"};
  }
}

template <ReduceType reduce_type>
void evalGeneric(const IPortableTensor *input, IPortableTensor *output,
                 const std::vector<int> &axes, bool keep_dims, nnfw::cker::Reduce &reduce_kernel)
{
  switch (input->data_type())
  {
    case OperandType::FLOAT32:
      return evalType<float>(input, output, axes, keep_dims, reduce_kernel, reduce_type);
    case OperandType::INT32:
      return evalType<int32_t>(input, output, axes, keep_dims, reduce_kernel, reduce_type);
    case OperandType::BOOL8:
      return evalType<bool>(input, output, axes, keep_dims, reduce_kernel, reduce_type);
    default:
      throw std::runtime_error{"Reduce(generic): unsupported data type"};
  }
}
} // namespace

ReduceLayer::ReduceLayer()
    : _input(nullptr), _output(nullptr), _reduceType(ReduceType::kAny), _axes(), _keep_dims(false),
      _reduce_kernel(new nnfw::cker::Reduce())
{
  // DO NOTHING
}

ReduceLayer::~ReduceLayer() = default;

void ReduceLayer::configure(const IPortableTensor *input, IPortableTensor *output,
                            ReduceType reduceType, const std::vector<int> &axes, bool keep_dims)
{
  _input = input;
  _output = output;
  _reduceType = reduceType;
  _axes = axes;
  _keep_dims = keep_dims;
}

void ReduceLayer::run()
{
  switch (_reduceType)
  {
    case ReduceType::kSum:
      evalGeneric<ReduceType::kSum>(_input, _output, _axes, _keep_dims, *_reduce_kernel);
      break;
    case ReduceType::kProd:
      evalGeneric<ReduceType::kProd>(_input, _output, _axes, _keep_dims, *_reduce_kernel);
      break;
    case ReduceType::kMax:
      evalGeneric<ReduceType::kMax>(_input, _output, _axes, _keep_dims, *_reduce_kernel);
      break;
    case ReduceType::kMin:
      evalGeneric<ReduceType::kMin>(_input, _output, _axes, _keep_dims, *_reduce_kernel);
      break;
    case ReduceType::kAny:
      evalGeneric<ReduceType::kAny>(_input, _output, _axes, _keep_dims, *_reduce_kernel);
      break;
    case ReduceType::kAll:
      evalGeneric<ReduceType::kAll>(_input, _output, _axes, _keep_dims, *_reduce_kernel);
      break;
    default:
      throw std::runtime_error{"ReduceSum: Unsupported reduce type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
