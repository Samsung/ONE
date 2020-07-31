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

#include "Dalgona.h"
#include "OperatorObserver.h"
#include "HDF5Importer.h"

#include <luci/Importer.h>

#include <pybind11/embed.h>

#include <fstream>
#include <iostream>

using Shape = luci_interpreter::Shape;
using DataType = luci_interpreter::DataType;

namespace py = pybind11;

namespace
{

/**
 * @brief  getTensorSize will return size in bytes
 */
template <typename NodeT> size_t getTensorSize(const NodeT *node)
{
  uint32_t tensor_size = loco::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

/**
 * @brief  verifyTypeShape checks the type and the shape of CircleInput
 *         This throws an exception if type or shape does not match
 */
bool verifyTypeShape(const luci::CircleInput *input_node, const DataType &dtype, const Shape &shape)
{
  // Type check
  if (dtype != input_node->dtype())
    throw std::runtime_error("Wrong input type.");

  if (shape.num_dims() != input_node->rank())
    throw std::runtime_error("Input rank mismatch.");

  for (uint32_t i = 0; i < shape.num_dims(); i++)
  {
    if (shape.dim(i) != input_node->dim(i).value())
      throw std::runtime_error("Input shape mismatch.");
  }
}

} // namespace

namespace dalgona
{

void Dalgona::initialize(const std::string &input_model_path)
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

  _observer = std::make_unique<OperatorObserver>(_interpreter.get());

  _interpreter->attachObserver(_observer.get());
}

void Dalgona::runAnalysis(const std::string &input_data_path, const std::string &analysis_path,
                          const std::string &analysis_args)
{
  py::object scope = py::module::import("__main__").attr("__dict__");
  _observer->importAnalysis(analysis_path, scope, analysis_args);

  HDF5Importer importer(input_data_path);

  bool is_raw_data = importer.isRawData();

  const auto num_records = importer.numRecords();
  if (num_records == 0)
    throw std::runtime_error("The input data file does not contain any record.");

  const auto input_nodes = loco::input_nodes(_module->graph());
  const auto num_inputs = input_nodes.size();

  for (int32_t record_idx = 0; record_idx < num_records; record_idx++)
  {
    if (num_inputs != importer.numInputs(record_idx))
      throw std::runtime_error("Wrong number of inputs.");

    if (record_idx % 100 == 0)
      std::cout << "Running " << record_idx << "'th data" << std::endl;

    for (int32_t input_idx = 0; input_idx < num_inputs; input_idx++)
    {
      const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
      assert(input_node->index() == input_idx);
      std::vector<char> input_data(getTensorSize(input_node));

      if (!is_raw_data)
      {
        DataType dtype;
        Shape shape(input_node->rank());
        importer.readTensor(record_idx, input_idx, &dtype, &shape, input_data.data());

        // Check the type and the shape of the input data is valid
        verifyTypeShape(input_node, dtype, shape);
      }
      else
      {
        // Skip type/shape check for raw data
        importer.readTensor(record_idx, input_idx, input_data.data());
      }

      _interpreter->writeInputTensor(input_node, input_data.data(), input_data.size());
    }

    _observer->startNetworkExecution(_module->graph());
    _interpreter->interpret();
    _observer->endNetworkExecution(_module->graph());
  }

  std::cout << "Finished executing " << num_records << "'th data" << std::endl;
  _observer->endAnalysis();
}

} // namespace dalgona
