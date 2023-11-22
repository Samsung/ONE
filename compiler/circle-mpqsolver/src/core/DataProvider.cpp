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

#include "DataProvider.h"

#include <luci/ImporterEx.h>
#include <luci/IR/CircleNodes.h>

using namespace mpqsolver::core;

using Shape = std::vector<loco::Dimension>;

namespace
{

// Check the type and the shape of input_node
// Throw an exception if type or shape does not match
void verifyTypeShape(const luci::CircleNode *input_node, const loco::DataType &dtype,
                     const Shape &shape)
{
  assert(input_node != nullptr); // FIX_CALLER_UNLESS

  // Type check
  if (dtype != input_node->dtype())
    throw std::runtime_error("Wrong input type.");

  if (shape.size() != input_node->rank())
    throw std::runtime_error("Input rank mismatch.");

  for (uint32_t i = 0; i < shape.size(); i++)
  {
    if (not(shape.at(i) == input_node->dim(i)))
      throw std::runtime_error("Input shape mismatch.");
  }
}

} // namespace

H5FileDataProvider::H5FileDataProvider(const std::string &h5file, const std::string &module_path)
  : _importer(h5file)
{
  _importer.importGroup("value");
  _is_raw_data = _importer.isRawData();

  luci::ImporterEx importerex;
  _module = importerex.importVerifyModule(module_path);
  if (_module.get() != nullptr)
  {
    _input_nodes = loco::input_nodes(_module.get()->graph());
  }
}

size_t H5FileDataProvider::numSamples() const { return _importer.numData(); }

uint32_t H5FileDataProvider::numInputs(uint32_t sample) const
{
  return static_cast<uint32_t>(_importer.numInputs(sample));
}

void H5FileDataProvider::getSampleInput(uint32_t sample, uint32_t input, InputData &data) const
{
  if (_is_raw_data)
  {
    _importer.readTensor(sample, input, data.data().data(), data.data().size());
  }
  else
  {
    loco::DataType dtype;
    Shape shape;
    _importer.readTensor(sample, input, &dtype, &shape, data.data().data(), data.data().size());

    // Check the type and the shape of the input data is valid
    auto input_node = loco::must_cast<luci::CircleNode *>(_input_nodes.at(input));
    verifyTypeShape(input_node, dtype, shape);
  }
}
