/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_GPU_CL_BACKEND_CONTEXT_H__
#define __ONERT_BACKEND_GPU_CL_BACKEND_CONTEXT_H__

#include <backend/BackendContext.h>
#include <util/ConfigSource.h>

#include <cl_common/BackendContext.h>

#include "ConstantInitializer.h"
#include "KernelGenerator.h"
#include "TensorBuilder.h"

#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class BackendContext
  : public onert::backend::cl_common::BackendContext<TensorBuilder, ConstantInitializer,
                                                     KernelGenerator>
{
public:
  BackendContext(const Backend *backend, ContextData &&data,
                 std::shared_ptr<TensorRegistry> tensor_registry = nullptr,
                 std::shared_ptr<TensorBuilder> tensor_builder = nullptr,
                 std::shared_ptr<ConstantInitializer> constant_initializer = nullptr,
                 std::shared_ptr<KernelGenerator> kernel_gen = nullptr)
    : onert::backend::cl_common::BackendContext<TensorBuilder, ConstantInitializer,
                                                KernelGenerator>(
        backend, std::move(data), tensor_registry, tensor_builder, constant_initializer, kernel_gen)
  {
    // DO NOTHING
  }

  ITensorRegistry *genTensors() override;
  FunctionMap genKernels() override;

protected:
  void registerTensorInfo(const ir::OperandIndex &ind, const ir::OperandInfo &info,
                          ir::Layout backend_layout) override;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_BACKEND_CONTEXT_H__
