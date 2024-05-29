/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <luci/Profile/CircleNodeOrigin.h>

#include "WeightOnlyFormatFileCreator.h"

namespace
{

constexpr uint16_t MAGIC_NUMBER = 429;
constexpr uint8_t SCHEMA_VERSION = 1;

} // namespace

namespace luci
{

/**
 * Calculate result buffer size
 **/
size_t WeightOnlyFormatFileCreator::calculateFileSize()
{
  assert(_model != nullptr);
  if (_model == nullptr)
    return 0;

  size_t result = 0;

  // 2 bytes for Magic Number
  result += 2;

  // 1 byte for Schema version
  result += 1;

  // 1 byte for Reserved field
  result += 1;

  // 4 bytes for number of tensors
  result += 4;

  CircleReader reader;
  if (!reader.parse(_model))
    return 0;

  // TODO: support multiple graphs
  if (reader.num_subgraph() != 1)
    return 0;

  reader.select_subgraph(0);

  auto tensors = reader.tensors();

  auto tensors_size = tensors.size();

  // tensors_size * 4 bytes for buffers offsets
  result += tensors_size * 4;

  for (uint32_t i = 0; i < tensors_size; ++i)
  {
    auto tensor = tensors[i];

    // check is const tensor
    auto data = _model->buffers()->operator[](tensor->buffer())->data();
    if (data == nullptr)
      continue;

    // check range of selected operators for training
    auto op_index = findOperatorIndex(i, reader);
    if (op_index == -1)
      continue;

    if (std::find(_ids.begin(), _ids.end(), op_index) == _ids.end())
      continue;

    result += data->size();
  }

  return result;
}

int32_t WeightOnlyFormatFileCreator::findOperatorIndex(const uint32_t tensor_index,
                                                       CircleReader &reader)
{
  auto operators = reader.operators();
  auto tensors = reader.tensors();

  for (uint32_t i = 0; i < operators.size(); ++i)
  {
    auto op = operators.at(i);
    auto inputs = op->inputs();

    for (auto input : *inputs)
    {
      if (input == tensor_index)
        return i;
    }
  }

  return -1;
}

/**
 * Calculate result buffer size
 **/
std::tuple<std::unique_ptr<char[]>, size_t> WeightOnlyFormatFileCreator::create()
{
  assert(_model != nullptr);
  if (_model == nullptr)
    return {nullptr, 0};

  size_t result_size = calculateFileSize();

  assert(result_size != 0);
  if (result_size == 0)
    return {nullptr, 0};

  auto result_buffer = std::make_unique<char[]>(result_size);

  // Point to start of the buffer
  char *cur_ptr = result_buffer.get();

  // Write MAGIC_NUMBER
  std::memcpy(cur_ptr, &MAGIC_NUMBER, sizeof(MAGIC_NUMBER));
  cur_ptr += 2;

  // Write SCHEMA_VERSION
  std::memcpy(cur_ptr, &SCHEMA_VERSION, sizeof(SCHEMA_VERSION));
  cur_ptr += 1;

  // Miss RESERVED field
  cur_ptr += 1;

  // Write number of buffers
  CircleReader reader;
  if (!reader.parse(_model))
    return {nullptr, 0};

  // TODO: support multiple graphs
  if (reader.num_subgraph() != 1)
    return {nullptr, 0};

  reader.select_subgraph(0);
  auto tensors = reader.tensors();
  uint32_t tensors_size = tensors.size();

  std::memcpy(cur_ptr, &tensors_size, sizeof(tensors_size));
  cur_ptr += 4;

  // Calculate offsets
  std::vector<uint32_t> offsets(tensors_size);
  uint32_t cur_offset = 8 + tensors_size * 4;
  for (uint32_t i = 0; i < tensors_size; ++i)
  {
    auto tensor = tensors[i];

    // Check is const tensor
    auto data = _model->buffers()->operator[](tensor->buffer())->data();
    if (data == nullptr)
    {
      offsets[i] = 0;
      continue;
    }

    // Check is index op for teaching
    auto op_index = findOperatorIndex(i, reader);
    if (op_index == -1)
      continue;
    if (std::find(_ids.begin(), _ids.end(), op_index) == _ids.end())
    {
      offsets[i] = 0;
      continue;
    }

    offsets[i] = cur_offset;
    cur_offset += data->size();
  }

  // Write offsets
  for (uint32_t i = 0; i < tensors_size; ++i)
  {
    uint32_t offset = offsets[i];
    std::memcpy(cur_ptr, &offset, sizeof(offset));
    cur_ptr += 4;
  }

  // Write buffers
  for (uint32_t i = 0; i < tensors_size; ++i)
  {
    uint32_t offset = offsets[i];

    if (offset == 0)
      continue;

    cur_ptr = result_buffer.get() + offset;
    auto tensor = tensors[i];

    // Check is const tensor
    auto data = _model->buffers()->operator[](tensor->buffer())->data();
    assert(data != nullptr);

    std::memcpy(cur_ptr, data->data(), data->size());
  }
#if 0
  // Dump wof
  std::cout << "DUMP:\n";
  // Magic NUM
  cur_ptr = result_buffer.get();
  char magic_num_char[] = {cur_ptr[0], cur_ptr[1]};
  uint16_t magic_num = *reinterpret_cast<uint16_t *>(magic_num_char);
  std::cout << "MAGIC_NUM = " << magic_num << "\n";

  // SCHEMA VER
  cur_ptr += 2;
  char shema_ver_char[] = {cur_ptr[0]};
  uint8_t shema_ver_num = *reinterpret_cast<uint8_t *>(shema_ver_char);
  std::cout << "SCHEMA VER = " << uint32_t(shema_ver_num) << "\n";

  // NUM TENSORS
  cur_ptr += 2;
  char num_tensors_char[] = {cur_ptr[0], cur_ptr[1], cur_ptr[2], cur_ptr[3]};
  uint32_t num_tensors_num = *reinterpret_cast<uint32_t *>(num_tensors_char);
  std::cout << "NUM TENSORS = " << num_tensors_num << "\n";

  // OFFSETS AND BUFFERS
  for (uint32_t i = 0; i < num_tensors_num; ++i)
  {
    cur_ptr += 4;
    char offsets_char[] = {cur_ptr[0], cur_ptr[1], cur_ptr[2], cur_ptr[3]};
    uint32_t offsets_num = *reinterpret_cast<uint32_t *>(offsets_char);
    std::cout << "OFFSET = " << offsets_num << "\n";
    if (offsets_num == 0)
      continue;

    std::cout << "DUMP WEIGTHS:\n ";
    auto tensor_size =
      _model->buffers()->operator[](tensors[i]->buffer())->data()->size() / sizeof(float);
    std::cout << "TENSOR SIZE = " << tensor_size << "\n";
    auto tmp_ptr = result_buffer.get() + offsets_num;
    for (uint32_t j = 0; j < tensor_size; ++j)
    {
      char w_char[] = {tmp_ptr[0], tmp_ptr[1], tmp_ptr[2], tmp_ptr[3]};
      float w_num = *reinterpret_cast<float *>(w_char);
      std::cout << j + 1 << " = " << w_num << "\n";
      tmp_ptr += 4;
    }
  }
#endif

  return {std::move(result_buffer), result_size};
}

} // namespace luci
