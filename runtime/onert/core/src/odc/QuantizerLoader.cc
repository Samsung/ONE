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

#include "QuantizerLoader.h"

#include <dlfcn.h>
#include <iostream>
#include <string>

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

QuantizerLoader &QuantizerLoader::instance()
{
  static QuantizerLoader singleton;
  return singleton;
}

int32_t QuantizerLoader::loadLibrary()
{
  if (get() != nullptr)
    return 0;

  const std::string quantize_so = std::string("libonert_odc") + SHARED_LIB_EXT;
  void *handle = dlopen(quantize_so.c_str(), RTLD_LAZY | RTLD_LOCAL);
  auto dlerror_msg = dlerror();

  if (handle == nullptr)
  {
    std::cerr << "Failed to load " << quantize_so << std::endl;
    std::cerr << dlerror_msg << std::endl;
    return 1;
  }

  {
    const char *factory_name = "create_quantizer";
    auto factory = (factory_t)dlsym(handle, factory_name);
    dlerror_msg = dlerror();

    if (factory == nullptr)
    {
      std::cerr << "QuantizerLoader: unable to find function " << factory_name << dlerror_msg
                << std::endl;
      dlclose(handle);
      return 1;
    }

    auto destroyer = (quantizer_destory_t)dlsym(handle, "destroy_quantizer");
    _quantizer = std::unique_ptr<IQuantizer, quantizer_destory_t>(factory(), destroyer);

    if (_quantizer == nullptr)
    {
      std::cerr << "QuantizerLoader: unable to create quantizer" << std::endl;
      return 1;
    }
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

int32_t QuantizerLoader::unloadLibrary()
{
  if (get() == nullptr)
    return 0;

  _quantizer.reset(nullptr);
  _dlhandle.reset(nullptr);

  return 0;
}

} // namespace odc
} // namespace onert
