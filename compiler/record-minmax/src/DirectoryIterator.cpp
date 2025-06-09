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

#include "DirectoryIterator.h"
#include "DataBuffer.h"
#include "Utils.h"

#include <luci/IR/Module.h>

#include <vector>
#include <string>
#include <cstring> // For memcpy

#include <dirent.h>

namespace record_minmax
{

DirectoryIterator::DirectoryIterator(const std::string &dir_path, luci::Module *module)
  : _dir_path(dir_path)
{
  _dir = opendir(dir_path.c_str());
  if (not _dir)
    throw std::runtime_error("Cannot open directory. Please check \"" + _dir_path +
                             "\" is a directory.\n");

  dirent *entry = nullptr;
  while ((entry = readdir(_dir)))
  {
    if (entry->d_type != DT_REG)
      continue;

    _entries.emplace_back(entry);
  }

  auto input_nodes = loco::input_nodes(module->graph());
  for (auto input_node : input_nodes)
  {
    const auto cnode = loco::must_cast<const luci::CircleInput *>(input_node);
    _input_nodes.emplace_back(cnode);
  }
}

DirectoryIterator::~DirectoryIterator()
{
  if (_dir)
    closedir(_dir);
};

bool DirectoryIterator::hasNext() const { return _curr_idx < _entries.size(); }

std::vector<DataBuffer> DirectoryIterator::next()
{
  auto entry = _entries.at(_curr_idx++);
  assert(entry); // FIX_ME_UNLESS

  // Get total input size
  uint32_t total_input_size = 0;
  for (auto input : _input_nodes)
  {
    const auto *input_node = loco::must_cast<const luci::CircleInput *>(input);
    total_input_size += getTensorSize(input_node);
  }

  const std::string filename = entry->d_name;

  // Read data from file to buffer
  // Assumption: For a multi-input model, the binary file should have inputs concatenated in the
  // same order with the input index.
  std::vector<char> input_data(total_input_size);
  readDataFromFile(_dir_path + "/" + filename, input_data, total_input_size);

  std::vector<DataBuffer> res;

  uint32_t offset = 0;
  for (auto input_node : _input_nodes)
  {
    DataBuffer buf;

    const auto input_size = getTensorSize(input_node);

    buf.data.resize(input_size);
    memcpy(buf.data.data(), input_data.data() + offset, input_size);

    offset += input_size;

    res.emplace_back(buf);
  }

  return res;
}

bool DirectoryIterator::check_type_shape() const { return false; }

} // namespace record_minmax
