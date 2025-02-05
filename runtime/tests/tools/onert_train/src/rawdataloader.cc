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
#include <cassert>

namespace onert_train
{
uint64_t getRawTensorSize(const std::vector<nnfw_tensorinfo> &infos)
{
  // NOTE The input_infos has already applied the batch_size.
  //      In order to know the size of the tensor stored in the actual file,
  //      the batch_size information must be excluded.
  uint64_t total = 0;
  for (uint32_t i = 0; i < infos.size(); ++i)
  {
    total += (bufsize_for(&infos[i]) / infos[i].dims[0]);
  }
  return total;
}
} // namespace onert_train

namespace onert_train
{

RawDataLoader::RawDataLoader(const std::string &input_file, const std::string &expected_file,
                             const std::vector<nnfw_tensorinfo> &input_infos,
                             const std::vector<nnfw_tensorinfo> &expected_infos)
  : DataLoader(input_infos, expected_infos)
{
  _input_file = std::ifstream(input_file, std::ios::binary);
  _expected_file = std::ifstream(expected_file, std::ios::binary);

  _input_file.seekg(0, std::ios::end);
  uint32_t input_file_size = _input_file.tellg();
  uint32_t input_data_length = input_file_size / getRawTensorSize(_input_infos);

  _expected_file.seekg(0, std::ios::end);
  uint32_t expected_file_size = _expected_file.tellg();
  uint32_t expected_data_length = expected_file_size / getRawTensorSize(_expected_infos);

  if (input_data_length != expected_data_length)
  {
    throw std::runtime_error("The length of input data and expected data does not match.");
  }

  _data_length = input_data_length;
}

std::tuple<Generator, uint32_t> RawDataLoader::loadData(const uint32_t batch_size, const float from,
                                                        const float to)
{
  assert(from >= 0.f && from <= 1.f);
  assert(to >= 0.f && to <= 1.f);
  assert(from <= to);

  int32_t split_size = _data_length * (to - from);
  int32_t split_start = _data_length * from;
  std::vector<uint32_t> input_origins(_input_infos.size());
  uint32_t start = 0;
  for (uint32_t i = 0; i < _input_infos.size(); ++i)
  {
    auto hwc_size = bufsize_for(&_input_infos[i]) / batch_size;
    input_origins.at(i) = start + (hwc_size * split_start);
    start += (hwc_size * _data_length);
  }

  std::vector<uint32_t> expected_origins(_expected_infos.size());
  start = 0;
  for (uint32_t i = 0; i < _expected_infos.size(); ++i)
  {
    auto hwc_size = bufsize_for(&_expected_infos[i]) / batch_size;
    expected_origins.at(i) = start + (hwc_size * split_start);
    start += (hwc_size * _data_length);
  }

  return std::make_tuple(
    [input_origins, expected_origins, this](uint32_t idx, std::vector<Allocation> &inputs,
                                            std::vector<Allocation> &expecteds) {
      for (uint32_t i = 0; i < _input_infos.size(); ++i)
      {
        auto bufsz = bufsize_for(&_input_infos[i]);
        _input_file.seekg(input_origins[i] + idx * bufsz, std::ios::beg);
        _input_file.read(reinterpret_cast<char *>(inputs[i].data()), bufsz);
      }
      for (uint32_t i = 0; i < _expected_infos.size(); ++i)
      {
        auto bufsz = bufsize_for(&_expected_infos[i]);
        _expected_file.seekg(expected_origins[i] + idx * bufsz, std::ios::beg);
        _expected_file.read(reinterpret_cast<char *>(expecteds[i].data()), bufsz);
      }
      return true;
    },
    split_size);
}

} // namespace onert_train
