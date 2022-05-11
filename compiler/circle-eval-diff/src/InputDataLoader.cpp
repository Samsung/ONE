/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "InputDataLoader.h"

#include <dio_hdf5/HDF5Importer.h>
#include <loco/IR/Graph.h>
#include <luci/IR/CircleNodes.h>

using DataType = loco::DataType;
using Shape = std::vector<loco::Dimension>;

namespace circle_eval_diff
{

// Check the type and the shape of CircleInput
void verifyTypeShape(const luci::CircleInput *input_node, const DataType &dtype, const Shape &shape)
{
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

} // namespace circle_eval_diff

namespace circle_eval_diff
{

HDF5Loader::HDF5Loader(const std::string &file_path, const std::vector<loco::Node *> &input_nodes)
  : _input_nodes{input_nodes}
{
  using HDF5Importer = dio::hdf5::HDF5Importer;

  _hdf5 = std::make_unique<HDF5Importer>(file_path);
  _hdf5->importGroup("value");
}

uint32_t HDF5Loader::size(void) const { return _hdf5->numData(); }

InputDataLoader::Data HDF5Loader::get(uint32_t data_idx) const
{
  Data data;
  data.resize(_input_nodes.size());

  for (uint32_t input_idx = 0; input_idx < _input_nodes.size(); input_idx++)
  {
    auto input_node = loco::must_cast<luci::CircleInput *>(_input_nodes.at(input_idx));
    assert(input_node->index() == input_idx);

    data.at(input_idx) = *createEmptyTensor(input_node).get();

    auto input_buffer = data.at(input_idx).buffer();
    if (_hdf5->isRawData())
    {
      _hdf5->readTensor(data_idx, input_idx, input_buffer);
    }
    else
    {
      DataType dtype;
      Shape shape;
      _hdf5->readTensor(data_idx, input_idx, &dtype, &shape, input_buffer);

      // Check the type and the shape of the input data is valid
      verifyTypeShape(input_node, dtype, shape);
    }
  }

  return data;
}

std::unique_ptr<InputDataLoader> makeDataLoader(const std::string &file_path,
                                                const InputFormat &format,
                                                const std::vector<loco::Node *> &input_nodes)
{
  switch (format)
  {
    case InputFormat::H5:
    {
      return std::make_unique<HDF5Loader>(file_path, input_nodes);
    }
    default:
      throw std::runtime_error{"Unsupported input format."};
  }
}

} // namespace circle_eval_diff
