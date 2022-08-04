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

#ifndef __ONERT_GPU_CL_OPEN_CL_FUNCTION_H__
#define __ONERT_GPU_CL_OPEN_CL_FUNCTION_H__

#include <exec/IFunction.h>

#include <vector>
#include <memory>

#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_command_queue.h"
#include "tensorflow/lite/delegates/gpu/cl/cl_operation.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{
class ClFunction : public ::onert::exec::IFunction
{
public:
  ClFunction(std::shared_ptr<tflite::gpu::cl::CreationContext> creation_context)
    : _creation_context(creation_context), _gpu_operations()
  {
  }

public:
  void add_operation(tflite::gpu::cl::ClOperation *gpu_operation)
  {
    _gpu_operations.push_back(gpu_operation);
  }

  void run() override
  {
    for (const auto gpu_operation : _gpu_operations)
    {
      if (!gpu_operation->AddToQueue(_creation_context->queue).ok())
      {
        throw std::runtime_error("Failed to AddToQueue.");
      }
    }
  }

  void prepare() override
  {
    for (const auto gpu_operation : _gpu_operations)
    {
      if (!gpu_operation->GetGpuOperation().AssembleCode(_creation_context->GetGpuInfo()).ok())
      {
        throw std::runtime_error("Failed to AssembleCode.");
      }
      if (!gpu_operation->Compile(*_creation_context).ok())
      {
        throw std::runtime_error("Failed to Compile.");
      }
      if (!gpu_operation->UpdateParams().ok())
      {
        throw std::runtime_error("Failed to UpdateParams.");
      }
      gpu_operation->GetGpuOperation().args_.ReleaseCPURepresentation();
    }
  }

private:
  std::shared_ptr<tflite::gpu::cl::CreationContext> _creation_context;
  std::vector<tflite::gpu::cl::ClOperation *> _gpu_operations;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_GPU_CL_OPEN_CL_FUNCTION_H__
