/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <iostream>

#include "TFLModel.h"

namespace tflite2circle
{

TFLModel::TFLModel(const std::string &path)
{
  _infile.open(path, std::ios::binary | std::ios::in);
  _valid = _infile.good();
}

const tflite::Model *TFLModel::load_model(void)
{
  assert(_valid == true);

  if (!_loaded)
  {
    _infile.seekg(0, std::ios::end);
    auto fileSize = _infile.tellg();
    _infile.seekg(0, std::ios::beg);
    _data.resize(fileSize);
    _infile.read(_data.data(), fileSize);
    _infile.close();
  }
  _loaded = true;

  return tflite::GetModel(_data.data());
}

bool TFLModel::verify_model(void)
{
  load_model();

  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(_data.data()), _data.size()};
  return tflite::VerifyModelBuffer(verifier);
}

} // namespace tflite2circle
