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

#ifndef __ONE_SERVICE_NPUD_CORE_DYNAMIC_LOADER_H__
#define __ONE_SERVICE_NPUD_CORE_DYNAMIC_LOADER_H__

#include "Backend.h"

#include <dlfcn.h>
#include <string>
#include <memory>

namespace npud
{
namespace core
{

using DLHandle = void *;

class DynamicLoader
{
public:
  DynamicLoader(const char *file, int flags = RTLD_LAZY);
  ~DynamicLoader();

  DynamicLoader(const DynamicLoader &) = delete;

  std::shared_ptr<Backend> getInstance();

private:
  DLHandle _handle;
  std::string _filepath;
  std::string _allocSymbol;
  std::string _deallocSymbol;
  std::shared_ptr<Backend> _backend;
};

} // namespace core
} // namespace npud

#endif // __ONE_SERVICE_NPUD_CORE_DYNAMIC_LOADER_H__
