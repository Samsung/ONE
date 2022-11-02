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

#include <tensorflow/lite/c/c_api.h>

namespace TFLiteRun
{

TensorDumper::TensorDumper()
{
  // DO NOTHING
}

void TensorDumper::addInputTensors(TfLiteInterpreter &interpreter)
{
  auto const input_count = TfLiteInterpreterGetInputTensorCount(&interpreter);
  for (int32_t idx = 0; idx < input_count; idx++)
  {
    const TfLiteTensor *tensor = TfLiteInterpreterGetInputTensor(&interpreter, idx);
    auto size = TfLiteTensorByteSize(tensor);
    std::vector<char> buffer;
    buffer.resize(size);
    memcpy(buffer.data(), TfLiteTensorData(tensor), size);
    _input_tensors.emplace_back(idx, std::move(buffer));
  }
}

void TensorDumper::addOutputTensors(TfLiteInterpreter &interpreter)
{
  auto const output_count = TfLiteInterpreterGetOutputTensorCount(&interpreter);
  for (int32_t idx = 0; idx < output_count; idx++)
  {
    const TfLiteTensor *tensor = TfLiteInterpreterGetOutputTensor(&interpreter, idx);
    auto size = TfLiteTensorByteSize(tensor);
    std::vector<char> buffer;
    buffer.resize(size);
    memcpy(buffer.data(), TfLiteTensorData(tensor), size);
    _output_tensors.emplace_back(idx, std::move(buffer));
  }
}

void TensorDumper::dump(const std::string &filename) const
{
  // TODO Handle file open/write error
  std::ofstream file(filename, std::ios::out | std::ios::binary);

  // Write number of tensors
  uint32_t num_tensors =
    static_cast<uint32_t>(_input_tensors.size()) + static_cast<uint32_t>(_output_tensors.size());
  file.write(reinterpret_cast<const char *>(&num_tensors), sizeof(num_tensors));

  // Write input tensor indices
  for (const auto &t : _input_tensors)
  {
    file.write(reinterpret_cast<const char *>(&t._index), sizeof(int));
  }

  // Write output tensor indices
  for (const auto &t : _output_tensors)
  {
    file.write(reinterpret_cast<const char *>(&t._index), sizeof(int));
  }

  // Write input data
  for (const auto &t : _input_tensors)
  {
    file.write(t._data.data(), t._data.size());
  }

  // Write output data
  for (const auto &t : _output_tensors)
  {
    file.write(t._data.data(), t._data.size());
  }

  file.close();
}

} // end of namespace TFLiteRun
