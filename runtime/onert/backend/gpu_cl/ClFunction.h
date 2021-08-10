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

#include "open_cl/kernels/GpuOperation.h"
#include "open_cl/ClCommandQueue.h"
#include "open_cl/Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class ClFunction : public ::onert::exec::IFunction
{
public:
  ClFunction() : _gpu_operations(), _creation_context() {}

public:
  void configure(std::shared_ptr<CreationContext> creation_context)
  {
    _creation_context = creation_context;
  }

  void add_operation(std::unique_ptr<GPUOperation> gpu_operation)
  {
    _gpu_operations.push_back(std::move(gpu_operation));
  }

  void run() override
  {
    for (const auto &gpu_operation : _gpu_operations)
    {
      if (!gpu_operation->AddToQueue(_creation_context->queue).ok())
      {
        throw std::runtime_error("Failed to AddToQueue.");
      }
    }
  }

  void prepare() override
  {
    for (const auto &gpu_operation : _gpu_operations)
    {
      if (!gpu_operation->Compile(*_creation_context).ok())
      {
        throw std::runtime_error("Failed to Compile.");
      }

      if (!gpu_operation->UpdateParams().ok())
      {
        throw std::runtime_error("Failed to UpdateParams.");
      }
    }
  }

private:
  std::vector<std::unique_ptr<GPUOperation>> _gpu_operations;
  std::shared_ptr<CreationContext> _creation_context;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_GPU_CL_OPEN_CL_FUNCTION_H__
