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

#include "KernelGenerator.h"
#include "TensorBuilder.h"
#include "open_cl/InferenceContext.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class BackendContext : public onert::backend::BackendContext
{
public:
  BackendContext(const Backend *backend, ContextData &&data,
                 std::shared_ptr<ITensorRegistry> tensor_registry = nullptr,
                 std::shared_ptr<TensorBuilder> tensor_builder = nullptr,
                 std::shared_ptr<KernelGenerator> kernel_gen = nullptr)
    : onert::backend::BackendContext(backend, std::move(data), tensor_registry),
      tensor_builder{tensor_builder}, kernel_gen{kernel_gen}
  {
  }

  ITensorRegistry *genTensors() override;
  FunctionMap genKernels() override;
  void planTensors();

public:
  std::shared_ptr<TensorBuilder> tensor_builder;
  std::shared_ptr<KernelGenerator> kernel_gen;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_BACKEND_CONTEXT_H__
