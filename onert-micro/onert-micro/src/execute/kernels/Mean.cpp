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
#include "core/OMCustomRuntimeData.h"

#include "execute/OMKernelExecutionBuilder.h"
#include "execute/OMRuntimeKernel.h"
#include "execute/OMUtils.h"

#include "OMStatus.h"
#include "PALReduceCommon.h"

using namespace onert_micro;
using namespace onert_micro::execute;

// ------------------------------------------------------------------------------------------------

namespace impl
{

template <typename T> OMStatus CircleMean(OMRuntimeKernel &rt_kernel)
{
  core::OMReduceDataContext<T> ctx(rt_kernel);

  bool is_ok = pal::Mean<T>(ctx);
  if (!is_ok)
  {
    return UnknownError;
  }

  return Ok;
}

} // namespace impl

// ------------------------------------------------------------------------------------------------

namespace onert_micro
{
namespace execute
{

OMStatus execute_kernel_CircleMean(const OMExecuteArgs &execute_args)
{
  core::OMRuntimeContext &runtime_context = execute_args.runtime_context;
  core::OMRuntimeStorage &runtime_storage = execute_args.runtime_storage;
  uint16_t op_index = execute_args.kernel_index;

  OMRuntimeKernel runtime_kernel;
  runtime_kernel.readKernel(op_index, runtime_context);
  runtime_kernel.getDataFromStorage(op_index, runtime_storage, runtime_context);

  constexpr static size_t kInputTensorIdx = 0;
  const circle::Tensor *input = runtime_kernel.inputs[kInputTensorIdx];

  switch (input->type())
  {
#ifndef DIS_FLOAT
    case circle::TensorType_FLOAT32:
      return impl::CircleMean<float>(runtime_kernel);
#endif // DIS_FLOAT

#ifndef DIS_QUANT
    case circle::TensorType_INT8:
      return impl::CircleMean<int8_t>(runtime_kernel);
#endif // DIS_QUANT

    case circle::TensorType_INT32:
    case circle::TensorType_INT64:
    default:
      assert(false && "Unsupported type");
      return UnsupportedType;
  }

  return Ok;
}

} // namespace execute
} // namespace onert_micro
