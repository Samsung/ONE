/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "VerifyFlatBuffers.h"

#include <foder/FileLoader.h>
#include <mio/tflite/schema_generated.h>

int VerifyFlatbuffers::run(const std::string &model_file)
{
  foder::FileLoader fileLoader{model_file};
  std::vector<char> modeldata = fileLoader.load();

  const uint8_t *data = reinterpret_cast<const uint8_t *>(modeldata.data());
  flatbuffers::Verifier verifier{data, modeldata.size()};

  if (!tflite::VerifyModelBuffer(verifier))
  {
    return -1;
  }

  return 0;
}
