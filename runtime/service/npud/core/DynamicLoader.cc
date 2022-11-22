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

#include "DynamicLoader.h"

#include "util/Logging.h"

namespace npud
{
namespace core
{

DynamicLoader::DynamicLoader(const char *file, int flags)
  : _handle(nullptr), _filepath(file), _allocSymbol("allocate"), _deallocSymbol("deallocate")
{
  if (!(_handle = dlopen(_filepath.c_str(), flags)))
  {
    VERBOSE(DynamicLoader) << "Fail to load " << _filepath << " module: " << dlerror() << std::endl;
    throw std::runtime_error("Fail to load " + _filepath + " module");
  }

  NpuAlloc alloc;
  NpuDealloc dealloc;

  alloc = reinterpret_cast<NpuAlloc>(dlsym(_handle, _allocSymbol.c_str()));
  dealloc = reinterpret_cast<NpuDealloc>(dlsym(_handle, _deallocSymbol.c_str()));
  if (!alloc || !dealloc)
  {
    VERBOSE(DynamicLoader) << "Fail to load " << _filepath << " symbol: " << dlerror() << std::endl;
    dlclose(_handle);
    throw std::runtime_error("Fail to load " + _filepath + " module");
  }

  _backend = std::shared_ptr<Backend>(alloc(), [dealloc](Backend *b) { dealloc(b); });
}

DynamicLoader::~DynamicLoader()
{
  // NOTE
  // The _backend shared_ptr must be explicitly deleted before
  // the dynamic library handle is released.
  _backend.reset();
  dlclose(_handle);
}

std::shared_ptr<Backend> DynamicLoader::getInstance() { return _backend; }

} // namespace core
} // namespace npud
