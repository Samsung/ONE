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

#include "HDF5Iterator.h"
#include "DataBuffer.h"
#include "Utils.h"

#include <luci/IR/Module.h>

#include <vector>
#include <string>

namespace record_minmax
{

HDF5Iterator::HDF5Iterator(const std::string &file_path, luci::Module *module)
  : _importer(file_path)
{
  try
  {
    _importer.importGroup("value");

    _is_raw_data = _importer.isRawData();

    _num_data = _importer.numData();
  }
  catch (const H5::Exception &e)
  {
    H5::Exception::printErrorStack();
    throw std::runtime_error("HDF5 error occurred during initialization.");
  }

  auto input_nodes = loco::input_nodes(module->graph());
  for (auto input_node : input_nodes)
  {
    const auto cnode = loco::must_cast<const luci::CircleInput *>(input_node);
    _input_nodes.emplace_back(cnode);
  }
}

bool HDF5Iterator::hasNext() const { return _curr_idx < _num_data; }

std::vector<DataBuffer> HDF5Iterator::next()
{
  std::vector<DataBuffer> res;

  try
  {
    for (int32_t input_idx = 0; input_idx < _importer.numInputs(_curr_idx); input_idx++)
    {
      DataBuffer buf;

      const auto input_node = _input_nodes.at(input_idx);
      const auto input_size = getTensorSize(input_node);
      buf.data.resize(input_size);

      if (check_type_shape())
      {
        _importer.readTensor(_curr_idx, input_idx, &buf.dtype, &buf.shape, buf.data.data(),
                             input_size);
      }
      else
      {
        _importer.readTensor(_curr_idx, input_idx, buf.data.data(), input_size);
      }

      res.emplace_back(buf);
    }
  }
  catch (const H5::Exception &e)
  {
    H5::Exception::printErrorStack();
    throw std::runtime_error("HDF5 error occurred during iteration.");
  }

  _curr_idx++; // move to the next index

  return res;
}

bool HDF5Iterator::check_type_shape() const
{
  // If it's raw data, we don't need to check type and shape
  return not _is_raw_data;
}

} // namespace record_minmax
