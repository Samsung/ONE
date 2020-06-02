/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "RecordMinMax.h"
#include "CircleExpContract.h"
#include "HDF5Importer.h"

#include <luci/Importer.h>
#include <luci/CircleExporter.h>

#include <fstream>
#include <stdexcept>

using Shape = luci_interpreter::Shape;
using DataType = luci_interpreter::DataType;

namespace
{

template <typename NodeT> size_t getTensorSize(const NodeT *node)
{
  uint32_t tensor_size = loco::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

} // namespace

namespace record_minmax
{

void RecordMinMax::initialize(const std::string &input_model_path)
{
  // Load model from the file
  std::ifstream fs(input_model_path, std::ifstream::binary);
  if (fs.fail())
  {
    throw std::runtime_error("Cannot open model file \"" + input_model_path + "\".\n");
  }
  std::vector<char> model_data((std::istreambuf_iterator<char>(fs)),
                               std::istreambuf_iterator<char>());
  _module = luci::Importer().importModule(circle::GetModel(model_data.data()));

  if (_module == nullptr)
  {
    throw std::runtime_error("ERROR: Failed to load '" + input_model_path + "'");
  }

  // Initialize interpreter
  _interpreter = std::make_unique<luci_interpreter::Interpreter>(_module.get());

  // TODO: Attach observer to the interpreter
}

void RecordMinMax::profileData(const std::string &input_data_path)
{
  HDF5Importer importer(input_data_path);

  const auto num_records = importer.numRecords();
  const auto input_nodes = loco::input_nodes(_module->graph());
  const auto num_inputs = input_nodes.size();

  for (int32_t i = 0; i < num_records; i++)
  {
    if (num_inputs != importer.numInputs(i))
      throw std::runtime_error("Wrong number of inputs.");

    for (int32_t j = 0; j < num_inputs; j++)
    {
      const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[j]);
      DataType dtype;
      std::vector<char> input_data(getTensorSize(input_node));
      importer.read(i, j, &dtype, input_data.data());

      if (dtype != input_node->dtype())
        throw std::runtime_error("Wrong input type.");

      // TODO: Input data is copied twice (file -> buffer (input_data) -> interpreter inputs)
      //       We can redcue the copy by directly writing data from file to interpreter inputs
      _interpreter->writeInputTensor(input_node, input_data.data(), input_data.size());
    }

    _interpreter->interpret();
  }

  // TODO: Determine the final min/max for each activation
  //       E.g., using clipping, averaging
}

void RecordMinMax::saveModel(const std::string &output_model_path)
{
  // TODO: Write min/max data to activation tensors in CircleNodes

  // Export to output Circle file
  luci::CircleExporter exporter;
  CircleExpContract contract(_module.get(), output_model_path);

  if (!exporter.invoke(&contract))
  {
    throw std::runtime_error("ERROR: Failed to export '" + output_model_path + "'");
  }
}

} // namespace record_minmax
