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

#include <foder/FileLoader.h>

#include "TFLModel.h"

namespace tflite2circle
{

TFLModel::TFLModel(const std::string &path)
{
  foder::FileLoader file_loader{path};
  _data = file_loader.load();

  // verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(_data.data()), _data.size()};
  if (!tflite::VerifyModelBuffer(verifier))
  {
    throw std::runtime_error("Failed to verify tflite");
  }
}

const tflite::Model *TFLModel::get_model(void) { return tflite::GetModel(_data.data()); }

} // namespace tflite2circle
