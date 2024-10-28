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

#include "ListFileIterator.h"
#include "DataBuffer.h"
#include "Utils.h"

#include <luci/IR/Module.h>

#include <vector>
#include <fstream>
#include <sstream> // For std::stringstream

namespace
{

// Return a string with no whitespace from both ends
std::string trim(std::string s)
{
  // Trim left side
  s.erase(s.begin(),
          std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));

  // Trim right side
  s.erase(
    std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
    s.end());

  return s;
}

// Return a vector of strings after splitting by space
std::vector<std::string> parse_line(const std::string &line)
{
  auto trimmed = trim(line);
  std::stringstream ss(trimmed);

  std::vector<std::string> res;

  std::string filename;
  while (getline(ss, filename, ' '))
  {
    res.emplace_back(filename);
  }
  return res;
}

} // namespace

namespace record_minmax
{

ListFileIterator::ListFileIterator(const std::string &input_path, luci::Module *module)
{
  std::ifstream input_file(input_path);
  if (input_file.fail())
    throw std::runtime_error("Cannot open file \"" + input_path + "\".\n");

  auto input_nodes = loco::input_nodes(module->graph());
  for (auto input_node : input_nodes)
  {
    const auto cnode = loco::must_cast<const luci::CircleInput *>(input_node);
    _input_nodes.emplace_back(cnode);
  }

  std::string record;
  while (getline(input_file, record))
  {
    _lines.emplace_back(record);
  }

  if (_lines.size() == 0)
    throw std::runtime_error("The input data file does not contain any record.");
}

bool ListFileIterator::hasNext() const { return _curr_idx < _lines.size(); }

std::vector<DataBuffer> ListFileIterator::next()
{
  const auto line = _lines.at(_curr_idx++);

  const auto file_names = parse_line(line);

  std::vector<DataBuffer> res;

  // Space-separated input files are written in a single line
  // This is the recommended way to write the list file
  if (file_names.size() == _input_nodes.size())
  {
    for (uint32_t i = 0; i < file_names.size(); i++)
    {
      DataBuffer buf;
      {
        const auto file_name = file_names.at(i);
        const auto input_node = _input_nodes.at(i);
        const auto input_size = getTensorSize(input_node);

        buf.data.resize(input_size);

        readDataFromFile(file_name, buf.data, input_size);
      }

      res.emplace_back(buf);
    }
  }
  else
  {
    // Must have a single file in one line (inputs are concatenated)
    if (file_names.size() != 1)
      throw std::runtime_error(
        "Wrong number of inputs are given. Model has " + std::to_string(_input_nodes.size()) +
        " inputs, but list file gives " + std::to_string(file_names.size()) + " inputs.");

    // Read data from file to buffer
    // Assumption: For a multi-input model, the binary file should have inputs concatenated in the
    // same order with the input index.
    // NOTE This is a legacy way to support multiple inputs.
    DataBuffer buf;
    {
      // Get total input size
      uint32_t total_input_size = 0;
      for (auto input_node : _input_nodes)
      {
        total_input_size += getTensorSize(input_node);
      }

      buf.data.resize(total_input_size);

      readDataFromFile(file_names.at(0), buf.data, total_input_size);
    }

    res.emplace_back(buf);
  }

  return res;
}

bool ListFileIterator::check_type_shape() const { return false; }

} // namespace record_minmax
