/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __NNKIT_BACKEND_PLUGIN_H__
#define __NNKIT_BACKEND_PLUGIN_H__

#include <nnkit/CmdlineArguments.h>
#include <nnkit/Backend.h>

#include <string>
#include <memory>

namespace nnkit
{

class BackendPlugin
{
public:
  typedef std::unique_ptr<Backend> (*Entry)(const CmdlineArguments &);

public:
  BackendPlugin(void *handle, Entry entry) : _handle{handle}, _entry{entry}
  {
    // DO NOTHING
  }

public:
  // Copy is not allowed to avoid double close
  BackendPlugin(const BackendPlugin &) = delete;
  BackendPlugin(BackendPlugin &&);

public:
  ~BackendPlugin();

public:
  std::unique_ptr<Backend> create(const CmdlineArguments &args) const;

private:
  void *_handle;
  Entry _entry;
};

std::unique_ptr<BackendPlugin> make_backend_plugin(const std::string &path);

} // namespace nnkit

#endif // __NNKIT_BACKEND_PLUGIN_H__
