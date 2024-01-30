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
#include "odc/CodegenManager.h"
#include "util/Utils.h"

#include <mutex>

namespace onert
{
namespace odc
{

bool CodegenManager::codegen(const char *target, CodegenPreference pref)
{
  if (target == nullptr)
    throw std::runtime_error("Target string is not set");

  if (_export_model_path.empty())
    throw std::runtime_error("Export model path is not set");

  if (_model_path.empty())
    throw std::runtime_error("Model path does not exist");

  // codegen function is thread-unsafe
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);

  auto &codegen_loader = CodegenLoader::instance();
  codegen_loader.loadLibrary(target);
  const auto code_generator = codegen_loader.get();
  // TODO Use compile preference
  UNUSED_RELEASE(pref);
  const auto result = code_generator->codegen(_model_path.c_str(), _export_model_path.c_str());
  codegen_loader.unloadLibrary();

  return (result == 0);
}

} // namespace odc
} // namespace onert
