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

#include "loader/ModelLoader.h"

#include "loader/ILoader.h"

#include <dlfcn.h>

namespace onert
{
namespace loader
{

std::unique_ptr<ir::Model> loadModel(const std::string &filename, const std::string &type)
{
  // Custom loader library name should be lib<type>_loader.so
  std::string libname = "lib" + type + "_loader.so";

  // Open custom loader library
  void *handle = dlopen(libname.c_str(), RTLD_LAZY);
  if (!handle)
    throw std::runtime_error("Failed to open " + type + " loader");

  // Get custom loader create function
  using create_func_t = ILoader *(*)();
  auto create_fn = reinterpret_cast<create_func_t>(dlsym(handle, "onert_loader_create"));
  if (!create_fn)
    throw std::runtime_error("Failed to find loader create function");

  // Get custom loader destroy function
  using destroy_func_t = void (*)(ILoader *);
  auto destroy_fn = reinterpret_cast<destroy_func_t>(dlsym(handle, "onert_loader_destroy"));
  if (!destroy_fn)
    throw std::runtime_error("Failed to find loader destroy function");

  // Create custom loader
  auto loader = create_fn();
  if (!loader)
    throw std::runtime_error("Failed to find loader create function");

  // Load model
  auto model = loader->loadFromFile(filename);

  // Destroy custom loader
  destroy_fn(loader);

  // Close custom loader library
  //
  // NOTE:
  //  It assumes that custom loader will not be used frequently on runtime session.
  //  If custom loader is used frequently, it should not close custom loader library and
  //  save handler to reuse it.
  dlclose(handle);

  if (model)
    return model;

  throw std::runtime_error("Failed to load model " + filename);
}

} // namespace loader
} // namespace onert
