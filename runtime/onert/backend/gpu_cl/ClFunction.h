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

#ifndef __ONERT_GPU_CL_CL_FUNCTION_H__
#define __ONERT_GPU_CL_CL_FUNCTION_H__

#include <exec/IFunction.h>
#include <memory>

#include "../open_cl/kernels/GpuOperation.h"
#include "../open_cl/ClCommandQueue.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class ClFunction : public ::onert::exec::IFunction
{
public:
  ClFunction() : _gpu_operation(), _creation_context() {}

public:
  void configure(std::unique_ptr<GPUOperation> gpu_operation,
                 std::shared_ptr<CreationContext> creation_context)
  {
    _gpu_operation = std::move(gpu_operation);
    _creation_context = std::move(creation_context);
  }

  void run() override { _gpu_operation->AddToQueue(_creation_context->queue); }

  void prepare() override
  {
    _gpu_operation->Compile(*_creation_context);
    _gpu_operation->UpdateParams();
  }

private:
  std::unique_ptr<GPUOperation> _gpu_operation;
  std::shared_ptr<CreationContext> _creation_context;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_GPU_CL_CL_FUNCTION_H__
