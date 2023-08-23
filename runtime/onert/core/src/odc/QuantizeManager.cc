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
#include "odc/QuantizeManager.h"

#include <iostream>
#include <mutex>

namespace onert
{
namespace odc
{

bool QuantizeManager::quantize()
{
  // Compile function is thread-unsafe
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);

  auto &quantize_loader = QuantizerLoader::instance();
  if (quantize_loader.loadLibrary() != 0)
    return false;

  auto const extension_pos = _model_path.find(".circle");
  if (extension_pos == std::string::npos)
  {
    std::cerr << "Input model isn't .circle." << std::endl;
    return false;
  }

  auto const qstring = std::string("_quantized_") + (_is_q16 ? "q16" : "q8");

  if (_export_model_path.empty())
    _export_model_path = _model_path.substr(0, extension_pos) + qstring + ".circle";

  auto quantizer = quantize_loader.get();
  auto result = quantizer->quantize(_model_path.c_str(), _export_model_path.c_str(), _is_q16);

  // TODO Unload quantize library to reduce memory usage

  return (result == 0);
}

} // namespace odc
} // namespace onert
