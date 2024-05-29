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

#include "MinMaxData.h"

#include <iostream>

namespace onert
{
namespace exec
{

RawMinMaxDumper::RawMinMaxDumper(const std::string &filename) : _filename(filename) {}

void RawMinMaxDumper::dump(const exec::IOMinMaxMap &input_minmax,
                           const exec::OpMinMaxMap &op_minmax) const
{
  // Find file is already exist for modifying
  auto file = std::fopen(_filename.c_str(), "rb+");
  uint32_t runs = 1;
  if (!file)
  {
    // If file is not exist, create new file
    file = std::fopen(_filename.c_str(), "wb+");
    if (!file)
      throw std::runtime_error{"Failed to open minmax file: " + _filename};
  }

  // Read run count
  std::fseek(file, 0, SEEK_SET);
  auto read_size = std::fread(&runs, sizeof(uint32_t), 1, file);
  if (read_size == 1)
    runs++;

  // TODO Verify file size

  // Overwrite run count
  std::fseek(file, 0, SEEK_SET);
  std::fwrite(&runs, sizeof(uint32_t), 1, file);

  // Go to end of file to append new data
  std::fseek(file, 0, SEEK_END);

  uint32_t input_count = input_minmax.size();
  uint32_t op_count = op_minmax.size();

  // Write op_count and input_count
  std::fwrite(&op_count, sizeof(uint32_t), 1, file);
  std::fwrite(&input_count, sizeof(uint32_t), 1, file);

  // For each op
  for (auto &&elem : op_minmax)
  {
    const uint32_t model_idx = 0;
    const uint32_t subg_idx = elem.first.first.value();
    const uint32_t op_idx = elem.first.second.value();

    // Write model/subg/op index
    std::fwrite(&model_idx, sizeof(uint32_t), 1, file);
    std::fwrite(&subg_idx, sizeof(uint32_t), 1, file);
    std::fwrite(&op_idx, sizeof(uint32_t), 1, file);

    // Write min/max
    std::fwrite(elem.second.data, sizeof(float), 2, file);
  }

  // For each input
  for (auto &&elem : input_minmax)
  {
    const uint32_t model_idx = 0;
    const uint32_t subg_idx = elem.first.first.value();
    const uint32_t input_idx = elem.first.second.value();

    // Write model/subg/input index
    std::fwrite(&model_idx, sizeof(uint32_t), 1, file);
    std::fwrite(&subg_idx, sizeof(uint32_t), 1, file);
    std::fwrite(&input_idx, sizeof(uint32_t), 1, file);

    // Write min/max
    std::fwrite(elem.second.data, sizeof(float), 2, file);
  }

  std::fclose(file);
}

} // namespace exec
} // namespace onert
