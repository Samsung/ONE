/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CodegenLoader.h"

#include <dlfcn.h>
#include <iostream>
#include <memory>

static const char *SHARED_LIB_EXT =
#if defined(__APPLE__) && defined(__MACH__)
  ".dylib";
#else
  ".so";
#endif

namespace onert
{
namespace odc
{

CodegenLoader &CodegenLoader::instance()
{
  static CodegenLoader singleton;
  return singleton;
}

void CodegenLoader::loadLibrary(const char *target)
{
  if (get() != nullptr)
    return;

  const std::string codegen_so = "lib" + std::string{target} + SHARED_LIB_EXT;
  void *handle = dlopen(codegen_so.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (handle == nullptr)
  {
    throw std::runtime_error("CodegenLoader: " + std::string{dlerror()});
  }

  const char *compile_impl_func_name = "generate_tvn_file";
  const auto compile = (codegen_t)dlsym(handle, compile_impl_func_name);
  if (compile == nullptr)
  {
    const std::string dlerror_msg = dlerror();
    dlclose(handle);
    throw std::runtime_error("CodegenLoader: " + dlerror_msg);
  }

  _codegen = compile;

  // Save backend handle (avoid warning by handle lost without dlclose())
  _dlhandle = std::unique_ptr<void, dlhandle_destroy_t>{
    handle, [filename = codegen_so](void *h) {
      if (dlclose(h))
        throw std::runtime_error("CodegenLoader: Failed to unload backend " + filename);
    }};
}

void CodegenLoader::unloadLibrary()
{
  if (get() == nullptr)
    return;

  // _codegen.reset(nullptr);
  _dlhandle.reset(nullptr);
}

int CodegenLoader::codegen(const char *in, const char *out)
{
  if (get() == nullptr)
    return -1;

  return _codegen(in, out);
}

} // namespace odc
} // namespace onert
