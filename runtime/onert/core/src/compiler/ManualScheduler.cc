/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ManualScheduler.h"
#include "ir/OpCode.h"
#include "ir/Operations.Include.h"
#include "backend/Backend.h"
#include "backend/IConfig.h"
#include "compiler/BackendManager.h"
#include "util/ConfigSource.h"
#include "util/logging.h"
#include "misc/string_helpers.h"

namespace onert::compiler
{

ManualScheduler::ManualScheduler(const std::vector<const backend::Backend *> &backends,
                                 const compiler::CompilerOptions &options)
  : _backends{backends}, _options{options}
{
}

std::unique_ptr<BackendResolver> ManualScheduler::schedule(const ir::Graph &graph)
{
  const auto &manual_options = _options.manual_scheduler_options;
  auto backend_resolver = std::make_unique<compiler::BackendResolver>();

  // This fallback order will be used in case that manual backend mapping is unavailable
  std::vector<const backend::Backend *> backend_order;
  for (auto &&backend_id : _options.backend_list)
  {
    auto backend = resolveBackend(backend_id);
    if (backend)
      backend_order.push_back(backend);
  }
  if (backend_order.size() == 0)
    throw std::runtime_error{"No loaded backends available."};

  // 1. Backend per operation type
  std::unordered_map<ir::OpCode, backend::Backend *> op_type_map;
  for (const auto &[op_code, backend_name] : manual_options.opcode_to_backend)
  {
    op_type_map.emplace(op_code, BackendManager::get().get(backend_name));
  }

  graph.operations().iterate([&](const ir::OperationIndex &index, const ir::IOperation &operation) {
    auto itr = op_type_map.find(operation.opcode());
    if (itr != op_type_map.end())
    {
      backend_resolver->setBackend(index, itr->second);
    }
  });

  // 2. Backend per operation index
  for (const auto &[key, val] : manual_options.index_to_backend)
  {
    try
    {
      graph.operations().at(key); // Check if exist, or this will throw
      backend_resolver->setBackend(key, BackendManager::get().get(val));
    }
    catch (...)
    {
      VERBOSE(ManualScheduler) << "Invalid value while OperationIndex to Backend mapping : @" << key
                               << " -> \"" << val << "\"" << std::endl;
    }
  }

  // 3. Fallback - backend priority order
  std::unordered_map<const backend::Backend *, std::unique_ptr<backend::ValidatorBase>> validators;
  for (auto &&backend : backend_order)
  {
    // Skip train backend because it's not supporting validator
    // TODO: Remove this condition when train backend supports validator
    if (backend->config()->id() == "train")
      continue;

    validators.emplace(backend, backend->validator(graph));
  }

  graph.operations().iterate([&](const ir::OperationIndex &index, const ir::IOperation &op) {
    if (!backend_resolver->hasBackend(index))
    {
      for (auto backend : backend_order)
      {
        // Use train backend if existed
        // On training mode, we should use train backend only for all operations
        // TODO: Remove this condition when train backend supports validator
        if (backend->config()->id() == "train")
        {
          backend_resolver->setBackend(index, backend);
          break;
        }

        if (validators[backend]->supported(op))
        {
          backend_resolver->setBackend(index, backend);
          break;
        }
      }
      if (!backend_resolver->hasBackend(index))
        throw std::runtime_error{"No backend found for operation @" +
                                 std::to_string(index.value())};
    }
  });

  // Dump final assignment
  WHEN_LOG_ENABLED(backend_resolver->iterate(
    [&](const ir::OperationIndex &index, const backend::Backend &backend) {
      VERBOSE(ManualScheduler) << "backend for " << index << ": " << backend.config()->id()
                               << std::endl;
    }));

  return backend_resolver;
}

const backend::Backend *ManualScheduler::resolveBackend(std::string_view id,
                                                        const backend::Backend *fallback)
{
  // Ensure if the backend is available in the current backend context
  const backend::Backend *backend = BackendManager::get().get(id);
  if (!backend || std::find(_backends.begin(), _backends.end(), backend) == _backends.end())
  {
    backend = fallback;
  }
  return backend;
}

} // namespace onert::compiler
