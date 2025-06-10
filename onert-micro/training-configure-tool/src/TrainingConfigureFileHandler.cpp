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

#include "TrainingConfigureFileHandler.h"

#include <fstream>
#include <vector>
#include <cstring>

namespace
{

constexpr uint16_t MAGIC_NUMBER = 29;
constexpr uint8_t SCHEMA_VERSION = 1;

void writeTrainConfigFileDataIntoBuffer(
  const training_configure_tool::TrainConfigFileData &train_data, std::vector<char> &buffer)
{
  const auto &train_op_indexes_with_ranks = train_data.trainable_op_indexes_with_ranks;

  // Resize to calculated size
  // 8 Bytes - handler + U16 indexes + U8 ranks
  auto buffer_size = 8 + train_op_indexes_with_ranks.size() * (sizeof(uint16_t) + sizeof(uint8_t));
  buffer.resize(buffer_size);

  // Point to start of the buffer
  char *cur_ptr = buffer.data();

  // Write MAGIC_NUMBER
  std::memcpy(cur_ptr, &MAGIC_NUMBER, sizeof(MAGIC_NUMBER));
  cur_ptr += 2;

  // Write SCHEMA_VERSION
  std::memcpy(cur_ptr, &SCHEMA_VERSION, sizeof(SCHEMA_VERSION));
  cur_ptr += 1;

  // Miss RESERVED field
  cur_ptr += 1;

  // Write number of layers
  auto layers_num = static_cast<int32_t>(train_op_indexes_with_ranks.size());
  std::memcpy(cur_ptr, &layers_num, sizeof(layers_num));
  cur_ptr += 4;

  // Write trainable layers positions
  for (const auto &p : train_op_indexes_with_ranks)
  {
    auto cur_layer_pos = p.first;
    std::memcpy(cur_ptr, &cur_layer_pos, sizeof(cur_layer_pos));
    cur_ptr += 2;
  }
  // Write code to define train rank of the trainable operation
  for (const auto &p : train_op_indexes_with_ranks)
  {
    const auto cur_layer_pos = static_cast<uint8_t>(p.second);
    std::memcpy(cur_ptr, &cur_layer_pos, sizeof(cur_layer_pos));
    cur_ptr += 1;
  }
}

} // namespace

void training_configure_tool::readDataFromFile(const std::string &filename, char *data,
                                               size_t data_size, size_t start_position)
{
  std::streampos start = start_position;

  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");

  fs.seekg(start);

  if (fs.read(data, data_size).fail())
    throw std::runtime_error("Failed to read data from file \"" + filename + "\".\n");
  fs.close();
}

void training_configure_tool::writeDataToFile(const std::string &filename, const char *data,
                                              size_t data_size)
{
  std::ofstream fs(filename, std::ofstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.write(data, data_size).fail())
  {
    throw std::runtime_error("Failed to write data to file \"" + filename + "\".\n");
  }
}

training_configure_tool::DataBuffer training_configure_tool::readFile(const char *path)
{
  std::ifstream file(path, std::ios::binary | std::ios::in);
  if (!file.good())
  {
    std::string errmsg = "Failed to open file";
    throw std::runtime_error(errmsg.c_str());
  }

  file.seekg(0, std::ios::end);
  auto fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // reserve capacity
  DataBuffer model_data(fileSize);

  // read the data
  file.read(model_data.data(), fileSize);
  if (file.fail())
  {
    std::string errmsg = "Failed to read file";
    throw std::runtime_error(errmsg.c_str());
  }

  return model_data;
}

onert_micro::OMStatus
training_configure_tool::createResultFile(const TrainConfigFileData &train_data,
                                          const char *save_path)
{
  std::vector<char> buffer;

  writeTrainConfigFileDataIntoBuffer(train_data, buffer);

  // Open or create file
  // Note: if the file existed, it will be overwritten
  std::ofstream out_file(save_path, std::ios::binary | std::ios::trunc);
  if (not out_file.is_open())
    return onert_micro::UnknownError;

  // Write data
  out_file.write(buffer.data(), static_cast<long>(buffer.size()));

  // Close file
  out_file.close();

  return onert_micro::Ok;
}

onert_micro::OMStatus
training_configure_tool::createResultData(const TrainConfigFileData &train_data,
                                          std::vector<char> &result_buffer)
{
  writeTrainConfigFileDataIntoBuffer(train_data, result_buffer);

  return onert_micro::Ok;
}
