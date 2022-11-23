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

#include "DevManager.h"
#include "util/Logging.h"

#include <dirent.h>

namespace npud
{
namespace core
{

DevManager::DevManager()
{
  const auto env = util::getConfigString(util::config::DEVICE_MODULE_PATH);
  _module_dir = std::move(env);
}

void DevManager::loadModules()
{
  VERBOSE(DevManager) << "load modules from " << _module_dir << std::endl;

  releaseModules();

  DIR *dir;
  struct dirent *entry;

  // NOTE
  // Return NULL(0) value when opendir or readdir error occurs.
  // NULL should be used instead of nullptr.
  dir = opendir(_module_dir.c_str());
  if (dir == NULL)
  {
    VERBOSE(DevManager) << "Fail to open module directory" << std::endl;
    return;
  }

  while ((entry = readdir(dir)) != NULL)
  {
    std::string modulePath(entry->d_name);
    if (modulePath.find("npud_backend") == std::string::npos)
    {
      continue;
    }

    DynamicLoader *loader = nullptr;
    try
    {
      loader = new DynamicLoader(modulePath.c_str());
    }
    catch (const std::exception &e)
    {
      VERBOSE(DevManager) << e.what() << std::endl;
      continue;
    }

    std::unique_ptr<Device> dev = std::make_unique<Device>();
    dev->modulePath = std::move(modulePath);
    dev->loader = std::unique_ptr<DynamicLoader>(loader);

    _dev = std::move(dev);
    break;
  }

  closedir(dir);
}

void DevManager::releaseModules()
{
  if (_dev)
  {
    _dev.reset();
  }
}

std::shared_ptr<Backend> DevManager::getBackend()
{
  if (!_dev)
  {
    throw std::runtime_error("No backend device.");
  }
  return _dev->loader->getInstance();
}

int DevManager::createContext(int deviceId, int priority, NpuContext **npuContext)
{
  try
  {
    return getBackend()->createContext(deviceId, priority, npuContext);
  }
  catch (const std::exception &e)
  {
    VERBOSE(DevManager) << e.what() << std::endl;
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }
}

int DevManager::destroyContext(NpuContext *npuContext)
{
  try
  {
    return getBackend()->destroyContext(npuContext);
  }
  catch (const std::exception &e)
  {
    VERBOSE(DevManager) << e.what() << std::endl;
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }
}

int DevManager::registerModel(NpuContext *npuContext, const std::string &modelPath,
                              ModelID *modelId)
{
  try
  {
    return getBackend()->registerModel(npuContext, modelPath, modelId);
  }
  catch (const std::exception &e)
  {
    VERBOSE(DevManager) << e.what() << std::endl;
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }
}

int DevManager::unregisterModel(NpuContext *npuContext, ModelID modelId)
{
  try
  {
    return getBackend()->unregisterModel(npuContext, modelId);
  }
  catch (const std::exception &e)
  {
    VERBOSE(DevManager) << e.what() << std::endl;
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }
}

int DevManager::createRequest(NpuContext *npuContext, ModelID modelId, RequestID *requestId)
{
  try
  {
    return getBackend()->createRequest(npuContext, modelId, requestId);
  }
  catch (const std::exception &e)
  {
    VERBOSE(DevManager) << e.what() << std::endl;
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }
}

int DevManager::destroyRequest(NpuContext *npuContext, RequestID requestId)
{
  try
  {
    return getBackend()->destroyRequest(npuContext, requestId);
  }
  catch (const std::exception &e)
  {
    VERBOSE(DevManager) << e.what() << std::endl;
    return NPU_STATUS_ERROR_OPERATION_FAILED;
  }
}

} // namespace core
} // namespace npud
