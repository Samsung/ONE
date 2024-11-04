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

#include "Utils.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/DataTypeHelper.h>

#include <vector>
#include <string>
#include <fstream>

namespace record_minmax
{

void checkInputDimension(const luci::CircleInput *input)
{
  assert(input); // FIX_CALLER_UNLESS

  for (uint32_t i = 0; i < input->rank(); i++)
    if (!input->dim(i).known())
      throw std::runtime_error(input->name() + " has unknown dimension");

  if (numElements(input) == 0)
    throw std::runtime_error(input->name() + " is a zero-sized input");
}

uint32_t numElements(const luci::CircleNode *node)
{
  assert(node); // FIX_CALLER_UNLESS

  uint32_t num_elements = 1;
  for (uint32_t i = 0; i < node->rank(); i++)
  {
    if (not node->dim(i).known())
      throw std::runtime_error("Unknown dimension found in " + node->name());

    num_elements *= node->dim(i).value();
  }

  return num_elements;
}

size_t getTensorSize(const luci::CircleNode *node)
{
  assert(node); // FIX_CALLER_UNLESS

  uint32_t elem_size = luci::size(node->dtype());
  return numElements(node) * elem_size;
}

void readDataFromFile(const std::string &filename, std::vector<char> &data, size_t data_size)
{
  assert(data.size() == data_size); // FIX_CALLER_UNLESS

  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.read(data.data(), data_size).fail())
    throw std::runtime_error("Failed to read data from file \"" + filename + "\".\n");
  if (fs.peek() != EOF)
    throw std::runtime_error("Input tensor size mismatches with \"" + filename + "\".\n");
}

} // namespace record_minmax
