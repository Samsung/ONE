/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Importer.h"
#include "luci/ImporterEx.h"

#include <foder/FileLoader.h>

#include <memory>
#include <iostream>
#include <stdexcept>

namespace luci
{

namespace
{

// limitation of current flatbuffers file size
inline constexpr uint64_t FLATBUFFERS_SIZE_MAX = 2147483648UL; // 2GB

} // namespace

ImporterEx::ImporterEx()
  : _error_handler{[](const std::exception &e) { std::cerr << e.what() << std::endl; }}
{
}

ImporterEx::ImporterEx(const std::function<void(const std::exception &)> &error_handler)
  : _error_handler{error_handler}
{
  if (!error_handler)
  {
    throw std::runtime_error{"The error handler passed to ImporterEx is invalid"};
  }
}

ImporterEx::ImporterEx(const GraphBuilderSource *source) : ImporterEx{} { _source = source; }

std::unique_ptr<Module> ImporterEx::importVerifyModule(const std::string &input_path) const
{
  foder::FileLoader file_loader{input_path};
  std::vector<char> model_data;

  try
  {
    model_data = file_loader.load();
  }
  catch (const std::runtime_error &err)
  {
    _error_handler(err);
    return nullptr;
  }

  auto data_data = reinterpret_cast<uint8_t *>(model_data.data());
  auto data_size = model_data.size();

  if (data_size < FLATBUFFERS_SIZE_MAX)
  {
    flatbuffers::Verifier verifier{data_data, data_size};
    if (!circle::VerifyModelBuffer(verifier))
    {
      _error_handler(std::runtime_error{"ERROR: Invalid input file '" + input_path + "'"});
      return nullptr;
    }
  }

  Importer importer(_source);
  return importer.importModule(data_data, data_size);
}

std::unique_ptr<Module> ImporterEx::importModule(const std::vector<char> &model_data) const
{
  auto data_data = reinterpret_cast<const uint8_t *>(model_data.data());
  auto data_size = model_data.size();

  Importer importer(_source);
  return importer.importModule(data_data, data_size);
}

} // namespace luci
