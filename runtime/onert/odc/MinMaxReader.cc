/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MinMaxReader.h"

#include <cstdio>
#include <stdexcept>

namespace
{

void inline readMMFile(void *ptr, size_t size, size_t count, FILE *fp, const std::string &err_msg)
{
  if (fread(ptr, size, count, fp) != count)
    throw std::runtime_error(err_msg);
}

} // namespace

namespace onert
{
namespace odc
{

MinMaxReader::MinMaxReader(const std::string &filepath) : _filepath(filepath)
{
  // DO NOTHING
}

// TODO: Handle multiple output
MinMaxVectors MinMaxReader::readOP(uint32_t model_idx, uint32_t subg_idx, uint32_t op_idx) const
{
  // Find file to read
  auto file = std::fopen(_filepath.c_str(), "rb");
  if (!file)
    throw std::runtime_error("Cannot open file: " + _filepath);

  // Check magic code and version
  // Match with runtime/onert/core/src/exec/MinMaxData.cc
  // TODO Use util to share code and version
  const uint32_t MAGIC_CODE = 0x4F4D4D44;
  const uint32_t VERSION = 1;
  {
    uint32_t read_magic_code = 0;
    uint32_t read_version = 0;
    if (std::fread(&read_magic_code, sizeof(uint32_t), 1, file) != 1 ||
        read_magic_code != MAGIC_CODE)
    {
      std::fclose(file);
      throw std::runtime_error{"MinMaxReader: Invalid magic code " + _filepath};
    }
    if (std::fread(&read_version, sizeof(uint32_t), 1, file) != 1 || read_version != VERSION)
    {
      std::fclose(file);
      throw std::runtime_error{"MinMaxReader: Invalid version " + _filepath};
    }
  }

  // Read num_run
  uint32_t num_run = 0;
  readMMFile(&num_run, sizeof(uint32_t), 1, file, "Cannot read num_run from file");

  MinMaxVectors mmv;
  float minmax[2];
  size_t data_size = sizeof(float) * 2 + sizeof(uint32_t) * 3;

  for (uint32_t r = 0; r < num_run; ++r)
  {
    // Read num of operations and num of inputs
    uint32_t num_op = 0;
    readMMFile(&num_op, sizeof(uint32_t), 1, file, "Cannot read num of operations");
    uint32_t num_input = 0;
    readMMFile(&num_input, sizeof(uint32_t), 1, file, "Cannot read num of inputs");

    // Find operation
    for (uint32_t i = 0; i < num_op; ++i)
    {
      uint32_t model_id_from_file = 0;
      uint32_t subg_idx_from_file = 0;
      uint32_t op_idx_from_file = 0;

      readMMFile(&model_id_from_file, sizeof(uint32_t), 1, file, "Cannot read model_id from file");
      readMMFile(&subg_idx_from_file, sizeof(uint32_t), 1, file, "Cannot read subg_idx from file");
      readMMFile(&op_idx_from_file, sizeof(uint32_t), 1, file, "Cannot read op_idx from file");

      if (model_id_from_file == model_idx && subg_idx_from_file == subg_idx &&
          op_idx_from_file == op_idx)
      {
        // Read minmax data
        readMMFile(minmax, sizeof(float), 2, file, "Cannot read minmax data from file");
        mmv.min_vector.emplace_back(minmax[0]);
        mmv.max_vector.emplace_back(minmax[1]);

        // Skip remain operation minmax data
        uint32_t remain_elem = num_op - i - 1;
        std::fseek(file, data_size * remain_elem, SEEK_CUR);
        break;
      }

      // Skip minmax data
      std::fseek(file, sizeof(float) * 2, SEEK_CUR);
    }

    // Skip input minmax data
    std::fseek(file, data_size * num_input, SEEK_CUR);
  }

  fclose(file);
  return mmv;
}

MinMaxVectors MinMaxReader::readInput(uint32_t model_idx, uint32_t subg_idx,
                                      uint32_t input_idx) const
{
  // Find file to read
  auto file = std::fopen(_filepath.c_str(), "rb");
  if (!file)
    throw std::runtime_error("Cannot open file: " + _filepath);

  // Read num_run
  uint32_t num_run = 0;
  readMMFile(&num_run, sizeof(uint32_t), 1, file, "Cannot read num_run from file");

  MinMaxVectors mmv;
  float minmax[2];
  size_t data_size = sizeof(float) * 2 + sizeof(uint32_t) * 3;

  for (uint32_t r = 0; r < num_run; ++r)
  {
    // Read num of operations and num of inputs
    uint32_t num_op = 0;
    readMMFile(&num_op, sizeof(uint32_t), 1, file, "Cannot read num of operations");
    uint32_t num_input = 0;
    readMMFile(&num_input, sizeof(uint32_t), 1, file, "Cannot read num of inputs");

    // Skip operation minmax data
    std::fseek(file, data_size * num_op, SEEK_CUR);

    // Find operation
    for (uint32_t i = 0; i < num_input; ++i)
    {
      uint32_t model_id_from_file = 0;
      uint32_t subg_idx_from_file = 0;
      uint32_t input_idx_from_file = 0;

      readMMFile(&model_id_from_file, sizeof(uint32_t), 1, file, "Cannot read model_id from file");
      readMMFile(&subg_idx_from_file, sizeof(uint32_t), 1, file, "Cannot read subg_idx from file");
      readMMFile(&input_idx_from_file, sizeof(uint32_t), 1, file,
                 "Cannot read input_idx from file");

      if (model_id_from_file == model_idx && subg_idx_from_file == subg_idx &&
          input_idx_from_file == input_idx)
      {
        // Read minmax data
        readMMFile(minmax, sizeof(float), 2, file, "Cannot read minmax data from file");
        mmv.min_vector.emplace_back(minmax[0]);
        mmv.max_vector.emplace_back(minmax[1]);

        // Skip remain input minmax data
        uint32_t remain_elem = num_input - i - 1;
        std::fseek(file, data_size * remain_elem, SEEK_CUR);
        break;
      }

      // Skip minmax data
      std::fseek(file, sizeof(float) * 2, SEEK_CUR);
    }
  }

  fclose(file);
  return mmv;
}

} // namespace odc
} // namespace onert
