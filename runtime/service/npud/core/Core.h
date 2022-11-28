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

#ifndef __ONE_SERVICE_NPUD_CORE_CORE_H__
#define __ONE_SERVICE_NPUD_CORE_CORE_H__

#include "DevManager.h"
#include "ContextManager.h"

#include <vector>
#include <string>

namespace npud
{
namespace core
{

// TODO Define error status

class Core
{
public:
  Core() noexcept;
  ~Core() noexcept = default;

  Core(const Core &) = delete;
  Core &operator=(const Core &) = delete;

  void init();
  void deinit();

  int getAvailableDeviceList(std::vector<std::string> &list) const;
  int createContext(int deviceId, int priority, ContextID *contextId) const;
  int destroyContext(ContextID contextId) const;
  int createNetwork(ContextID contextId, const std::string &modelPath, ModelID *modelId) const;
  int destroyNetwork(ContextID contextId, ModelID modelId) const;
  int createRequest(ContextID contextId, ModelID modelId, RequestID *requestId) const;
  int destroyRequest(ContextID contextId, RequestID requestId) const;

private:
  std::unique_ptr<DevManager> _devManager;
  std::unique_ptr<ContextManager> _contextManager;
};

} // namespace core
} // namespace npud

#endif // __ONE_SERVICE_NPUD_CORE_CORE_H__
