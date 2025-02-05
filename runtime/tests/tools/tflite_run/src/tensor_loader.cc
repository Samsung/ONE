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

#include "tensor_loader.h"

#include <assert.h>

#include <cstring>
#include <fstream>

#include "misc/tensor/Shape.h"

namespace TFLiteRun
{

TensorLoader::TensorLoader(TfLiteInterpreter &interpreter)
  : _interpreter(interpreter), _raw_data(nullptr)
{
}

void TensorLoader::loadDumpedTensors(const std::string &filename)
{
  // TODO Handle file open/read error
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  uint32_t num_tensors = 0;
  file.read(reinterpret_cast<char *>(&num_tensors), sizeof(num_tensors));

  int tensor_indices_raw[num_tensors];
  file.read(reinterpret_cast<char *>(tensor_indices_raw), sizeof(tensor_indices_raw));

  _raw_data = std::unique_ptr<float[]>(new float[file_size]);
  file.read(reinterpret_cast<char *>(_raw_data.get()), file_size);
  file.close();

  size_t read_bytes = loadInputTensorsFromRawData();
  read_bytes += loadOutputTensorsFromRawData();

  // The file size and total output tensor size must match
  assert(file_size ==
         sizeof(num_tensors) + sizeof(tensor_indices_raw) + read_bytes * sizeof(float));
}

void TensorLoader::loadRawInputTensors(const std::string &filename)
{
  // TODO Handle file open/read error
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  _raw_data = std::unique_ptr<float[]>(new float[file_size]);
  file.read(reinterpret_cast<char *>(_raw_data.get()), file_size);
  file.close();

  size_t read_bytes = loadInputTensorsFromRawData();

  // The file size and total output tensor size must match
  assert(file_size == read_bytes * sizeof(float));
}

size_t TensorLoader::loadInputTensorsFromRawData()
{
  size_t offset = 0;
  auto const input_count = TfLiteInterpreterGetInputTensorCount(&_interpreter);
  for (auto idx = 0; idx < input_count; idx++)
  {
    const TfLiteTensor *tensor = TfLiteInterpreterGetInputTensor(&_interpreter, idx);

    // Convert tensor shape to `Shape` from `tensor->dims`
    nnfw::misc::tensor::Shape shape(TfLiteTensorNumDims(tensor));
    for (int32_t d = 0; d < TfLiteTensorNumDims(tensor); d++)
    {
      shape.dim(d) = TfLiteTensorDim(tensor, d);
    }

    float *base = _raw_data.get() + offset;

    assert(TfLiteTensorByteSize(tensor) % sizeof(float) == 0);
    offset += (TfLiteTensorByteSize(tensor) / sizeof(float));

    _input_tensor_map.emplace(idx, nnfw::tflite::TensorView<float>(shape, base));

    memcpy(TfLiteTensorData(tensor), reinterpret_cast<const void *>(base),
           TfLiteTensorByteSize(tensor));
  }

  return offset;
}

size_t TensorLoader::loadOutputTensorsFromRawData()
{
  size_t offset = 0;
  auto const output_count = TfLiteInterpreterGetOutputTensorCount(&_interpreter);
  for (auto idx = 0; idx < output_count; idx++)
  {
    const TfLiteTensor *tensor = TfLiteInterpreterGetOutputTensor(&_interpreter, idx);

    // Convert tensor shape to `Shape` from `tensor->dims`
    nnfw::misc::tensor::Shape shape(TfLiteTensorNumDims(tensor));
    for (int32_t d = 0; d < TfLiteTensorNumDims(tensor); d++)
    {
      shape.dim(d) = TfLiteTensorDim(tensor, d);
    }

    float *base = _raw_data.get() + offset;

    assert(TfLiteTensorByteSize(tensor) % sizeof(float) == 0);
    offset += (TfLiteTensorByteSize(tensor) / sizeof(float));

    _output_tensor_map.emplace(idx, nnfw::tflite::TensorView<float>(shape, base));

    memcpy(TfLiteTensorData(tensor), reinterpret_cast<const void *>(base),
           TfLiteTensorByteSize(tensor));
  }

  return offset;
}

const nnfw::tflite::TensorView<float> &TensorLoader::getOutput(int tensor_idx) const
{
  auto found = _output_tensor_map.find(tensor_idx);
  assert(found != _output_tensor_map.end());
  return found->second;
}

} // end of namespace TFLiteRun
