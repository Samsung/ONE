/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "rawdataloader.h"
#include "nnfw_util.h"

#include <iostream>
#include <stdexcept>
#include <numeric>

namespace onert_train
{

Generator RawDataLoader::loadData(const std::string &input_file, const std::string &expected_file,
                                  const std::vector<nnfw_tensorinfo> &input_infos,
                                  const std::vector<nnfw_tensorinfo> &expected_infos,
                                  const uint32_t data_length, const uint32_t batch_size)
{
  auto size_accumulator = [](const unsigned int &a, const nnfw_tensorinfo &b) {
    return a + bufsize_for(&b);
  };

  std::vector<uint32_t> input_origins(input_infos.size());
  uint32_t start = 0;
  for (uint32_t i = 0; i < input_infos.size(); ++i)
  {
    input_origins.at(i) = start;
    start += (bufsize_for(&input_infos[i]) * data_length);
  }

  std::vector<uint32_t> expected_origins(expected_infos.size());
  start = 0;
  for (uint32_t i = 0; i < expected_infos.size(); ++i)
  {
    expected_origins.at(i) = start;
    start += (bufsize_for(&expected_infos[i]) * data_length);
  }

  try
  {
    _input_file = std::ifstream(input_file, std::ios::ate | std::ios::binary);
    _expected_file = std::ifstream(expected_file, std::ios::ate | std::ios::binary);
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    std::exit(-1);
  }

  return [input_origins, expected_origins, &input_infos, &expected_infos, batch_size,
          this](uint32_t idx, std::vector<Allocation> &inputs, std::vector<Allocation> &expecteds) {
    for (uint32_t i = 0; i < input_infos.size(); ++i)
    {
      auto bufsz = bufsize_for(&input_infos[i]);
      _input_file.seekg(input_origins[i] + idx * batch_size * bufsz, std::ios::beg);
      for (uint32_t j = 0; j < batch_size; ++j)
      {
        _input_file.read(reinterpret_cast<char *>(inputs[i * batch_size + j].data()), bufsz);
      }
    }
    for (uint32_t i = 0; i < expected_infos.size(); ++i)
    {
      auto bufsz = bufsize_for(&expected_infos[i]);
      _expected_file.seekg(expected_origins[i] + idx * batch_size * bufsz, std::ios::beg);
      for (uint32_t j = 0; j < batch_size; ++j)
      {
        _expected_file.read(reinterpret_cast<char *>(expecteds[i * batch_size + j].data()), bufsz);
      }
    }
    return true;
  };
}

} // namespace onert_train
