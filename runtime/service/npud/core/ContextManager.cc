/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ContextManager.h"

#include <algorithm>
#include <util/Logging.h>

namespace npud
{
namespace core
{

ContextManager::ContextManager() noexcept {}

ContextManager::~ContextManager() noexcept { _contexts.clear(); }

void ContextManager::newContext(NpuContext *npuContext, ContextID *contextId)
{
  auto context = std::make_unique<Context>();
  // TODO Consider the possibility of reusing the same address.
  context->contextId = reinterpret_cast<ContextID>(context.get());
  context->npuContext = npuContext;
  *contextId = context->contextId;
  _contexts.emplace_back(std::move(context));

  this->listContexts();
}

void ContextManager::deleteContext(ContextID contextId)
{
  const auto iter =
    std::remove_if(_contexts.begin(), _contexts.end(),
                   [&](std::unique_ptr<Context> &c) { return c->contextId == contextId; });
  if (iter == _contexts.end())
  {
    return;
  }

  _contexts.erase(iter, _contexts.end());

  this->listContexts();
}

void ContextManager::listContexts()
{
#ifdef DEBUG
  VERBOSE(ContextManager) << "Size: " << _contexts.size() << std::endl;
  for (const auto &context : _contexts)
  {
    VERBOSE(ContextManager) << "==========================" << std::endl;
    VERBOSE(ContextManager) << "contextId: " << context->contextId << std::endl;
  }
  VERBOSE(ContextManager) << "==========================" << std::endl;
#endif
}

const std::vector<std::unique_ptr<Context>>::iterator
ContextManager::getContext(ContextID contextId)
{
  const auto iter =
    std::find_if(_contexts.begin(), _contexts.end(),
                 [&](std::unique_ptr<Context> &c) { return c->contextId == contextId; });
  return iter;
}

NpuContext *ContextManager::getNpuContext(ContextID contextId)
{
  const auto iter = getContext(contextId);
  if (iter == _contexts.end())
  {
    return nullptr;
  }

  return iter->get()->npuContext;
}

} // namespace core
} // namespace npud
