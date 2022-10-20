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
#include "DynamicLoader.h"

#include <dirent.h>
#include <algorithm>
#include <util/Logging.h>
#include <util/ConfigSource.h>

namespace npud
{
namespace core
{

#define DEFAULT_DEVICE_PATH "/usr/lib/npud/devices"

DevManager::DevManager() : _devs{}
{
  const auto env = util::getConfigString(util::config::DEVICE_MODULE_PATH);
  _module_dir = std::move(env);
}

DevManager::~DevManager() {}

void DevManager::loadModules(void)
{
  VERBOSE(DevManager) << "load modules from " << _module_dir << std::endl;

  static uint64_t _devInstance = 0;

  if (_devs.size() > 0)
  {
    _devs.clear();
  }

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
    dev->devId = _devInstance++;
    dev->modulePath = std::move(modulePath);
    dev->device = nullptr; // TODO Implement
    dev->loader = std::unique_ptr<DynamicLoader>(loader);

    _devs.emplace_back(std::move(dev));
  }

  closedir(dir);

  listModules();

  std::string version;
  getDevice(0)->loader->getInstance()->getVersion(version);
  getDevice(0)->loader->getInstance()->createContext(nullptr, 0, 0, nullptr);
}

void DevManager::listModules(void)
{
  for (auto &dev : _devs)
  {
    VERBOSE(DevManager) << "==========================" << std::endl;
    VERBOSE(DevManager) << "devId: " << dev->devId << std::endl;
    VERBOSE(DevManager) << "modulePath: " << dev->modulePath << std::endl;
  }
  VERBOSE(DevManager) << "==========================" << std::endl;
}

void DevManager::releaseModules(void)
{
  for (auto &dev : _devs)
  {
    dev->loader.reset();
  }
  if (_devs.size() > 0)
  {
    _devs.clear();
  }
}

Device *DevManager::getDevice(DevID id)
{
  auto iter = std::find_if(_devs.begin(), _devs.end(),
                           [&](std::unique_ptr<Device> &d) { return d->devId == id; });
  if (iter == _devs.end())
  {
    throw std::runtime_error("DevID is not valid.");
  }
  return iter->get();
}

std::shared_ptr<Backend> DevManager::getBackend(DevID id)
{
  return this->getDevice(id)->loader->getInstance();
}

} // namespace core
} // namespace npud
