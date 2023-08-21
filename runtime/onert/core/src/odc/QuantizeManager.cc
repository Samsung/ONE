/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "odc/QuantizeManager.h"
#include "odc/Quantize.h"

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

QuantizeManager &QuantizeManager::instance()
{
  static QuantizeManager singleton;
  return singleton;
}

Quantize *QuantizeManager::get() const { return _default_quantize.get(); }

int32_t QuantizeManager::loadLibrary()
{
  if (get() != nullptr)
    return 0;

  // TODO: Determine the name of plug-in library
  //
  // Option 1: Use predefined hardcoded name for on-device compiler implementation plug-in
  //           e.g) libodc.so
  // Option 2: Search plug-in in pre-defined paths, then load the shared library name.
  //           e.g) $ ls /usr/lib/onert/plug-ins
  //                  libone-quantize.so
  //                Then, call loadCompiler("one-quantize");

  const auto findPlugin = []() { return "onert_odc"; }; // TODO: rename to generic name.
  const std::string id = findPlugin();
  const std::string quantize_so = "lib" + id + SHARED_LIB_EXT;
  void *handle = dlopen(quantize_so.c_str(), RTLD_LAZY | RTLD_LOCAL);

  if (handle == nullptr)
  {
    std::cerr << "Failed to load " << quantize_so << std::endl;
    std::cerr << dlerror() << std::endl;
    return 1;
  }

  {
    const char *quantize_impl_func_name = "quantize";
    const auto quantize = (Quantize::quantize_t)dlsym(handle, quantize_impl_func_name);
    if (quantize == nullptr)
    {
      std::cerr << "QuantizeManager: unable to find function " << quantize_impl_func_name
                << dlerror() << std::endl;
      dlclose(handle);
      return 1;
    }

    _default_quantize = std::make_unique<Quantize>(quantize);
  }

  // Save quantize library handle (avoid warning by handle lost without dlclose())
  // clang-format off
  _dlhandle = std::unique_ptr<void, dlhandle_destroy_t>{handle, [filename = quantize_so](void *h) {
    if (dlclose(h) != 0)
      std::cerr << "Failed to unload backend " << filename << std::endl;
  }};
  // clang-format on

  return 0;
}

} // namespace odc
} // namespace onert
