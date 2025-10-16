/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "execute/OMUtils.h"
#include "execute/OMKernelExecutionBuilder.h"
#include "OMStatus.h"
#include "execute/OMRuntimeKernel.h"
#include "core/OMUtils.h"

#include "core/OMRuntimeShape.h"
#include "PALReduceCommon.h"

using namespace onert_micro;
using namespace onert_micro::execute;

// ------------------------------------------------------------------------------------------------

namespace impl
{

template <typename T> OMStatus CircleReduceProd(OMRuntimeKernel &rt_kernel)
{
  constexpr static T kInitValue = 1;

  core::OMReduceDataContext<T> ctx(rt_kernel);
  pal::Reducer<T, pal::ReduceProductFn> reducer(ctx, kInitValue);

  bool is_ok = reducer.Reduce();
  if (!is_ok)
  {
    OM_LOG_AND_RETURN(UnknownError, "Unknown error encountered");
  }

  return Ok;
}

} // namespace impl

// ------------------------------------------------------------------------------------------------

namespace onert_micro
{
namespace execute
{

OMStatus execute_kernel_CircleReduceProd(const OMExecuteArgs &execute_args)
{
  core::OMRuntimeContext &rt_context = execute_args.runtime_context;
  core::OMRuntimeStorage &rt_storage = execute_args.runtime_storage;

  uint16_t op_index = execute_args.kernel_index;

  OMRuntimeKernel rt_kernel;
  rt_kernel.readKernel(op_index, rt_context);
  rt_kernel.getDataFromStorage(op_index, rt_storage, rt_context);

  constexpr static size_t kInputTensorIdx = 0;

  const circle::Tensor *input = rt_kernel.inputs[kInputTensorIdx];

  // TODO: Add ReducerOptions support

  switch (input->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
      return impl::CircleReduceProd<float>(rt_kernel);
#endif // DIS_FLOAT

#ifndef DIS_QUANT
    case circle::TensorType_INT8:
      return impl::CircleReduceProd<int8_t>(rt_kernel);
#endif // DIS_QUANT

    case circle::TensorType_INT32:
      return impl::CircleReduceProd<int32_t>(rt_kernel);

    case circle::TensorType_INT64:
      return impl::CircleReduceProd<int64_t>(rt_kernel);

    default:
      assert(false && "Unsupported type");
      OM_LOG_AND_RETURN(UnsupportedType, "Unsupported type encountered");
  }

  return Ok;
}

} // namespace execute
} // namespace onert_micro
