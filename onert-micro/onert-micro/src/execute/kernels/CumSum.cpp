/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "core/OMUtils.h"
#include "core/OMKernelData.h"

#include "execute/OMUtils.h"

#include "PALCumSum.h"

using onert_micro::OMStatus;

using namespace onert_micro::core;
using namespace onert_micro::execute;

// ------------------------------------------------------------------------------------------------

namespace impl
{

constexpr static uint32_t axisTensorIdx = 1;
constexpr static uint32_t inputTensorIdx = 0;
constexpr static uint32_t outputTensorIdx = 0;

template <typename T> OMStatus CircleCumSum(OMRuntimeKernel &rt_kernel)
{
  const circle::Tensor *axis = rt_kernel.inputs[axisTensorIdx];
  const circle::Tensor *input = rt_kernel.inputs[inputTensorIdx];
  const circle::Tensor *output = rt_kernel.outputs[outputTensorIdx];

  assert(axis != nullptr);
  assert(input != nullptr);
  assert(output != nullptr);

  const int32_t *axis_data = utils::castInputData<int32_t>(rt_kernel.inputs_data[axisTensorIdx]);
  const T *input_data = utils::castInputData<T>(rt_kernel.inputs_data[inputTensorIdx]);
  T *output_data = utils::castOutputData<T>(rt_kernel.outputs_data[outputTensorIdx]);

  assert(input_data != nullptr);
  assert(output_data != nullptr);

  OMRuntimeShape input_shape(input);

  const int32_t rank = input_shape.dimensionsCount();
  const int32_t axis_value = (axis_data != nullptr) ? *axis_data : 0;

  assert(rank >= 1);
  assert(axis_value >= 0);
  assert(axis_value < rank);

  auto options = rt_kernel.first_operator->builtin_options_as_CumsumOptions();

  assert(options != nullptr);

  CumSumParams params{};
  params.exclusive = options->exclusive();
  params.reverse = options->reverse();

  return pal::CumSum<T>(params, input_shape, input_data, axis_value, output_data);
}

} // namespace impl

// ------------------------------------------------------------------------------------------------

namespace onert_micro::execute
{

OMStatus execute_kernel_CircleCumSum(const OMExecuteArgs &execute_args)
{
  core::OMRuntimeContext &rt_context = execute_args.runtime_context;
  core::OMRuntimeStorage &rt_storage = execute_args.runtime_storage;

  uint16_t op_index = execute_args.kernel_index;

  OMRuntimeKernel rt_kernel;
  OMStatus status = rt_kernel.readKernel(op_index, rt_context);

  if (status != Ok)
    return status;

  const circle::Tensor *input = rt_kernel.inputs[impl::inputTensorIdx];
  assert(input != nullptr);

  status = rt_kernel.getDataFromStorage(op_index, rt_storage, rt_context);

  if (status != Ok)
    return status;

  switch (input->type())
  {
    case circle::TensorType_INT32:
      return impl::CircleCumSum<int32_t>(rt_kernel);

    case circle::TensorType_INT64:
      return impl::CircleCumSum<int64_t>(rt_kernel);

#ifndef DIS_FLOAT

    case circle::TensorType_FLOAT32:
      return impl::CircleCumSum<float>(rt_kernel);

#endif // DIS__FLOAT

    default:
      assert(false && "Unsupported type.");
      return UnsupportedType;
  }
}

} // namespace onert_micro::execute
