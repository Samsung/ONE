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

#include "Core.h"

#include <util/Logging.h>

namespace npud
{
namespace core
{

Core::Core() noexcept
  : _devManager(std::make_unique<DevManager>()), _contextManager(std::make_unique<ContextManager>())
{
  VERBOSE(Core) << "Constructor" << std::endl;
}

Core::~Core() noexcept { VERBOSE(Core) << "Destructor" << std::endl; }

void Core::init() { _devManager->loadModules(); }

void Core::deinit() { _devManager->releaseModules(); }

int Core::createContext(int deviceId, int priority, ContextID *contextId)
{
  VERBOSE(Core) << "createContext with " << deviceId << ", " << priority << std::endl;
  NpuContext *npuContext;
  int ret = _devManager->createContext(deviceId, priority, &npuContext);
  if (ret != NPU_STATUS_SUCCESS) {
    VERBOSE(Core) << "Fail to create dev context" << std::endl;
    // TODO Define CoreStatus
    return 1;
  }

  ContextID _contextId;
  _contextManager->newContext(npuContext, &_contextId);
  *contextId = _contextId;
  return 0;
}

int Core::destroyContext(ContextID contextId)
{
  VERBOSE(Core) << "destroyContext with " << contextId << std::endl;
  NpuContext *npuContext = _contextManager->getNpuContext(contextId);
  if (!npuContext)
  {
    VERBOSE(Core) << "Invalid context id" << std::endl;
    // TODO Define CoreStatus
    return 1;
  }

  int ret = _devManager->destroyContext(npuContext);
  if (ret != NPU_STATUS_SUCCESS) {
    VERBOSE(Core) << "Fail to destroy npu context" << std::endl;
  }

  _contextManager->deleteContext(contextId);
  return 0;
}

int Core::createNetwork(ContextID contextId, const std::string &modelPath, ModelID *modelId)
{
  VERBOSE(Core) << "createNetwork with " << contextId << ", " << modelPath << std::endl;
  NpuContext *npuContext = _contextManager->getNpuContext(contextId);
  if (!npuContext)
  {
    VERBOSE(Core) << "Invalid context id" << std::endl;
    // TODO Define CoreStatus
    return 1;
  }

  ModelID id;
  int ret = _devManager->registerModel(npuContext, modelPath, &id);
  if (ret != NPU_STATUS_SUCCESS) {
    VERBOSE(Core) << "Failed to register model: " << modelPath << std::endl;
    // TODO Define CoreStatus
    return 1;
  }

  *modelId = id;
  return 0;
}

} // namespace core
} // namespace npud
