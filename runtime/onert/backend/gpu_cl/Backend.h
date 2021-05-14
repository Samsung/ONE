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

#ifndef __ONERT_BACKEND_GPU_CL_BACKEND_H__
#define __ONERT_BACKEND_GPU_CL_BACKEND_H__

#include <backend/Backend.h>
#include <memory>

#include "BackendContext.h"
#include "Config.h"
#include "ClTensorRegistry.h"
#include "KernelGenerator.h"
#include "TensorManager.h"
#include "TensorBuilder.h"

#include "open_cl/Environment.h"
#include "open_cl/Status.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

class Backend : public ::onert::backend::Backend
{
public:
  Backend() : _config{std::make_shared<Config>()} {}

  std::shared_ptr<IConfig> config() const override { return _config; }

  std::unique_ptr<onert::backend::BackendContext> newContext(ContextData &&data) const override
  {
    const auto &graph = *data.graph;
    const auto &operands = data.graph->operands();
    auto context = std::make_unique<gpu_cl::BackendContext>(this, std::move(data));

    auto environment = std::make_unique<Environment>();
    if (!CreateEnvironment(environment.get()).ok())
    {
      return nullptr;
    }
    auto tm = createTensorManager(&environment->context());
    auto tr = std::make_shared<ClTensorRegistry<TensorManager>>(tm);

    InferenceContext::CreateInferenceInfo create_info;
    create_info.precision = environment->IsSupported(CalculationsPrecision::F16)
                              ? CalculationsPrecision::F16
                              : CalculationsPrecision::F32;
    create_info.storage_type = GetFastestStorageType(environment->device().GetInfo());
    create_info.hints.Add(ModelHints::kFastestInference);

    auto tb = std::make_shared<TensorBuilder>(operands, tm, create_info, environment->queue(),
                                              environment->GetDevicePtr());
    context->tensor_registry = tr;
    context->tensor_builder = tb;
    auto cc = std::make_shared<CreationContext>();
    cc->device = environment->GetDevicePtr();
    cc->context = &environment->context();
    cc->queue = environment->queue();
    cc->cache = environment->program_cache();
    context->kernel_gen = std::make_shared<KernelGenerator>(graph, tb, tr, cc);
    context->constant_initializer = std::make_shared<ConstantInitializer>(operands, tr);
    return context;
  }

private:
  std::shared_ptr<IConfig> _config;
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_BACKEND_H__
