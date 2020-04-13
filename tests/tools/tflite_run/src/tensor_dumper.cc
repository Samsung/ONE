/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tensor_dumper.h"

#include <fstream>
#include <iostream>
#include <cstring>

#include "tensorflow/lite/interpreter.h"

namespace TFLiteRun
{

TensorDumper::TensorDumper()
{
  // DO NOTHING
}

void TensorDumper::addTensors(tflite::Interpreter &interpreter, const std::vector<int> &indices)
{
  for (const auto &o : indices)
  {
    const TfLiteTensor *tensor = interpreter.tensor(o);
    int size = tensor->bytes;
    std::vector<char> buffer;
    buffer.resize(size);
    memcpy(buffer.data(), tensor->data.raw, size);
    _tensors.emplace_back(o, std::move(buffer));
  }
}

void TensorDumper::dump(const std::string &filename) const
{
  // TODO Handle file open/write error
  std::ofstream file(filename, std::ios::out | std::ios::binary);

  // Write number of tensors
  uint32_t num_tensors = static_cast<uint32_t>(_tensors.size());
  file.write(reinterpret_cast<const char *>(&num_tensors), sizeof(num_tensors));

  // Write tensor indices
  for (const auto &t : _tensors)
  {
    file.write(reinterpret_cast<const char *>(&t._index), sizeof(int));
  }

  // Write data
  for (const auto &t : _tensors)
  {
    file.write(t._data.data(), t._data.size());
  }

  file.close();
}

} // end of namespace TFLiteRun
