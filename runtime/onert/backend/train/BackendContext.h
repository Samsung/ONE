/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_BACKEND_CONTEXT_H__
#define __ONERT_BACKEND_TRAIN_BACKEND_CONTEXT_H__

#include <backend/BackendContext.h>

#include "ExternalContext.h"
#include "KernelGenerator.h"
#include "TensorBuilder.h"

namespace onert
{
namespace backend
{
namespace train
{

class BackendContext : public onert::backend::BackendContext
{
public:
  BackendContext(const Backend *backend, ContextData &&data,
                 std::shared_ptr<ITensorRegistry> tensor_registry = nullptr,
                 std::shared_ptr<TensorBuilder> tensor_builder = nullptr,
                 std::shared_ptr<KernelGenerator> kernel_gen = nullptr)
    : onert::backend::BackendContext(backend, std::move(data), tensor_registry),
      tensor_builder{tensor_builder}, kernel_gen{kernel_gen}, _external_context(new ExternalContext)
  {
  }

  ITensorRegistry *genTensors() override;
  backend::FunctionMap genKernels() override;

  std::shared_ptr<ExternalContext> external_context() { return _external_context; }

public:
  // TODO Make it private
  std::shared_ptr<TensorBuilder> tensor_builder;
  std::shared_ptr<KernelGenerator> kernel_gen;

private:
  // NOTE ruy context has a thread pool, and when multiple ruy contexts are created,
  //      the thread pool is also created in duplicate
  // TODO Create one ruy context for session
  std::shared_ptr<ExternalContext> _external_context;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_BACKEND_CONTEXT_H__
