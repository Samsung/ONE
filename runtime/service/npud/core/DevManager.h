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

#ifndef __ONE_SERVICE_NPUD_CORE_DEV_MANAGER_H__
#define __ONE_SERVICE_NPUD_CORE_DEV_MANAGER_H__

#include "Backend.h"
#include "DynamicLoader.h"

#include <vector>
#include <memory>

namespace npud
{
namespace core
{

using DevID = uint64_t;
struct Device
{
  DevID devId;
  std::string modulePath;
  std::unique_ptr<NpuDevice> device;
  std::unique_ptr<DynamicLoader> loader;
};

class DevManager
{
public:
  DevManager();
  ~DevManager();

  DevManager(const DevManager &) = delete;
  DevManager &operator=(const DevManager &) = delete;

  void loadModules(void);
  void releaseModules(void);
  std::shared_ptr<Backend> getBackend();

  int createContext(int deviceId, int priority, NpuContext **npuContext);
  int destroyContext(NpuContext *npuContext);

private:
  Device *getDevice(DevID id);
  void listModules(void);

private:
  std::vector<std::unique_ptr<Device>> _devs;
  std::string _module_dir;
  DevID _defaultId;
};

} // namespace core
} // namespace npud

#endif // __ONE_SERVICE_NPUD_CORE_DEV_MANAGER_H__
