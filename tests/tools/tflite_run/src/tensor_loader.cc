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

#include <fstream>

#include "misc/tensor/Shape.h"

namespace TFLiteRun
{

TensorLoader::TensorLoader(tflite::Interpreter &interpreter)
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
  std::vector<int> tensor_indices(tensor_indices_raw, tensor_indices_raw + num_tensors);

  _raw_data = std::unique_ptr<float[]>(new float[file_size]);
  file.read(reinterpret_cast<char *>(_raw_data.get()), file_size);
  file.close();

  size_t read_bytes = loadTensorsFromRawData(tensor_indices);

  // The file size and total output tensor size must match
  assert(file_size ==
         sizeof(num_tensors) + sizeof(tensor_indices_raw) + read_bytes * sizeof(float));
}

void TensorLoader::loadRawTensors(const std::string &filename,
                                  const std::vector<int> &tensor_indices)
{
  // TODO Handle file open/read error
  std::ifstream file(filename, std::ios::ate | std::ios::binary);
  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  _raw_data = std::unique_ptr<float[]>(new float[file_size]);
  file.read(reinterpret_cast<char *>(_raw_data.get()), file_size);
  file.close();

  size_t read_bytes = loadTensorsFromRawData(tensor_indices);

  // The file size and total output tensor size must match
  assert(file_size == read_bytes * sizeof(float));
}

size_t TensorLoader::loadTensorsFromRawData(const std::vector<int> &tensor_indices)
{
  size_t offset = 0;
  for (const auto &o : tensor_indices)
  {
    const TfLiteTensor *tensor = _interpreter.tensor(o);

    // Convert tensor shape to `Shape` from `tensor->dims`
    nnfw::misc::tensor::Shape shape(static_cast<size_t>(tensor->dims->size));
    for (int d = 0; d < tensor->dims->size; d++)
    {
      shape.dim(d) = tensor->dims->data[d];
    }

    float *base = _raw_data.get() + offset;

    assert(tensor->bytes % sizeof(float) == 0);
    offset += (tensor->bytes / sizeof(float));

    _tensor_map.insert(std::make_pair(o, nnfw::tflite::TensorView<float>(shape, base)));
  }

  return offset;
}

const nnfw::tflite::TensorView<float> &TensorLoader::get(int tensor_idx) const
{
  auto found = _tensor_map.find(tensor_idx);
  assert(found != _tensor_map.end());
  return found->second;
}

} // end of namespace TFLiteRun
