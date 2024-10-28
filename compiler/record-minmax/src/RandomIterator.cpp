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

#include "RandomIterator.h"
#include "DataBuffer.h"
#include "Utils.h"

#include <luci/IR/Module.h>
#include <luci/IR/CircleNodes.h>

#include <vector>
#include <cstring>

namespace
{

std::vector<float> genRandomData(std::mt19937 &gen, uint32_t num_elements, float min, float max)
{
  std::uniform_real_distribution<float> dist(min, max);
  std::vector<float> input_data(num_elements);

  // Write random data
  {
    auto const generator = [&gen, &dist]() { return static_cast<float>(dist(gen)); };
    std::generate(begin(input_data), end(input_data), generator);
  }

  return input_data;
}

template <typename T>
std::vector<T> genRandomIntData(std::mt19937 &gen, uint32_t num_elements, T min, T max)
{
  std::uniform_int_distribution<T> dist(min, max);
  std::vector<T> input_data(num_elements);

  // Write random data
  {
    auto const generator = [&gen, &dist]() { return dist(gen); };
    std::generate(begin(input_data), end(input_data), generator);
  }

  return input_data;
}

} // namespace

namespace record_minmax
{

RandomIterator::RandomIterator(luci::Module *module)
{
  assert(module); // FIX_CALLER_UNLESS

  std::random_device rd;
  std::mt19937 _gen(rd());

  auto input_nodes = loco::input_nodes(module->graph());
  for (auto input_node : input_nodes)
  {
    const auto cnode = loco::must_cast<const luci::CircleInput *>(input_node);
    _input_nodes.emplace_back(cnode);
  }

  // Hardcoded
  _num_data = 3;
}

bool RandomIterator::hasNext() const { return _curr_idx < _num_data; }

std::vector<DataBuffer> RandomIterator::next()
{
  std::vector<DataBuffer> res;

  for (auto input_node : _input_nodes)
  {
    DataBuffer buf;

    const auto dtype = input_node->dtype();
    const auto num_elements = numElements(input_node);

    buf.data.resize(getTensorSize(input_node));

    switch (dtype)
    {
      case loco::DataType::FLOAT32:
      {
        const auto input_data = genRandomData(_gen, num_elements, -5, 5);
        const auto data_size = input_data.size() * sizeof(float);
        assert(buf.data.size() == data_size);
        memcpy(buf.data.data(), input_data.data(), data_size);
        break;
      }
      case loco::DataType::S32:
      {
        const auto input_data = genRandomIntData<int32_t>(_gen, num_elements, 0, 100);
        const auto data_size = input_data.size() * sizeof(int32_t);
        assert(buf.data.size() == data_size);
        memcpy(buf.data.data(), input_data.data(), data_size);
        break;
      }
      case loco::DataType::S64:
      {
        const auto input_data = genRandomIntData<int64_t>(_gen, num_elements, 0, 100);
        const auto data_size = input_data.size() * sizeof(int64_t);
        assert(buf.data.size() == data_size);
        memcpy(buf.data.data(), input_data.data(), data_size);
        break;
      }
      case loco::DataType::BOOL:
      {
        const auto input_data = genRandomIntData<uint8_t>(_gen, num_elements, 0, 1);
        const auto data_size = input_data.size() * sizeof(uint8_t);
        assert(buf.data.size() == data_size);
        memcpy(buf.data.data(), input_data.data(), data_size);
        break;
      }
      default:
        throw std::runtime_error("Unsupported datatype");
    }

    res.emplace_back(buf);
  }

  _curr_idx++; // move to the next index

  return res;
}

bool RandomIterator::check_type_shape() const { return false; }

} // namespace record_minmax
