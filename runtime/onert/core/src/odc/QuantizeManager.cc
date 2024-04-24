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

namespace
{

std::string inline getModelName(const std::string &model_path)
{
  auto extension_pos = model_path.find(".circle");
  auto model_dir_pos = model_path.find_last_of("/");
  if (model_dir_pos == std::string::npos)
    model_dir_pos = 0;
  else
    model_dir_pos++;
  return model_path.substr(model_dir_pos, extension_pos - model_dir_pos);
}

std::string inline getExportModelFile(const std::string &model_name, onert::odc::QuantizeType type)
{
  switch (type)
  {
    case onert::odc::QuantizeType::ODC_QTYPE_U8_ASYM:
      return model_name + ".q8.circle";
    case onert::odc::QuantizeType::ODC_QTYPE_I16_SYM:
      return model_name + ".q16.circle";
    case onert::odc::QuantizeType::ODC_QTYPE_WO_I8_SYM:
      return model_name + ".q8wo.circle";
    case onert::odc::QuantizeType::ODC_QTYPE_WO_I16_SYM:
      return model_name + ".q16wo.circle";
    default:
      throw std::runtime_error{"Not supported quantization type"};
  }

  throw std::runtime_error{"Not supported quantization type"};
}

} // namespace

namespace onert
{
namespace odc
{

bool QuantizeManager::quantize()
{
  if (_model_path.empty())
    return false;

  auto model_name = getModelName(_model_path);
  _export_model_path = _workspace_dir + "/" + getExportModelFile(model_name, _qtype);

  // Compile function is thread-unsafe
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);

  auto &quantize_loader = QuantizerLoader::instance();
  if (quantize_loader.loadLibrary() != 0)
    return false;

  auto quantizer = quantize_loader.get();

  quantizer->setMinMaxPath(_workspace_dir + "/minmax.bin");
  auto result = quantizer->quantize(_model_path.c_str(), _export_model_path.c_str(), _qtype);

  // TODO Unload quantize library to reduce memory usage

  return (result == 0);
}

} // namespace odc
} // namespace onert
