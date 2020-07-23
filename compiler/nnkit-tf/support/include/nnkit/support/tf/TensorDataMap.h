/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNKIT_SUPPORT_TF_TENSOR_DATA_MAP_H__
#define __NNKIT_SUPPORT_TF_TENSOR_DATA_MAP_H__

#include "nnkit/support/tftestinfo/ParsedTensor.h"

#include <tensorflow/c/c_api.h>

#include <stdexcept>
#include <memory>
#include <map>

namespace nnkit
{
namespace support
{
namespace tf
{

using nnkit::support::tftestinfo::ParsedTensor;

/**
 * @brief Class to map parsed tensor and memory for tensor values.
 *  For parsed tensor, this memory is used to fill input or output values of graph.
 */
class TensorDataMap
{
public:
  TensorDataMap()
  { /* empty */
  }

  uint8_t *allocate(const ParsedTensor *parsed_tensor)
  {
    auto it = _data_map.find(parsed_tensor);
    if (it != _data_map.end())
      throw std::runtime_error("Already allocated");

    int bytes = 0;
    if (parsed_tensor->isFloatTensor())
      bytes = sizeof(float);
    else
      throw std::runtime_error("Unsupported or wrong data type");

    uint64_t size = num_elements(parsed_tensor->shape()) * bytes;
    _data_map[parsed_tensor] = std::move(std::unique_ptr<uint8_t[]>(new uint8_t[size]));

    return _data_map[parsed_tensor].get();
  }

  uint8_t *data(const ParsedTensor *parsed_tensor)
  {
    auto it = _data_map.find(parsed_tensor);
    if (it == _data_map.end())
      throw std::runtime_error("Cannot find parsed tensor");

    return it->second.get();
  }

private:
  std::map<const ParsedTensor *, std::unique_ptr<uint8_t[]>> _data_map;
};

} // namespace tf
} // namespace support
} // namespace nnkit

#endif // __NNKIT_SUPPORT_TF_TENSOR_DATA_MAP_H__
