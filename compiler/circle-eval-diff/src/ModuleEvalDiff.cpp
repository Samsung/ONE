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

#include "ModuleEvalDiff.h"
#include "Tensor.h"

#include <luci_interpreter/Interpreter.h>
#include <dio_hdf5/HDF5Importer.h>

#include <string>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <fstream>

using Tensor = circle_eval_diff::Tensor;
using DataType = loco::DataType;
using Shape = std::vector<loco::Dimension>;
using HDF5Importer = dio::hdf5::HDF5Importer;

namespace
{

void writeDataToFile(const std::string &filename, const char *data, size_t data_size)
{
  std::ofstream fs(filename, std::ofstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.write(data, data_size).fail())
  {
    throw std::runtime_error("Failed to write data to file \"" + filename + "\".\n");
  }
}

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

// Return number of elements of the node.
uint32_t numElements(const luci::CircleNode *node)
{
  uint32_t num_elem = 1;
  for (uint32_t i = 0; i < node->rank(); ++i)
    num_elem *= node->dim(i).value();
  return num_elem;
}

// Return Tensor which has the same dtype and shape with node.
// Buffer does not have any data yet.
std::shared_ptr<Tensor> createEmptyTensor(const luci::CircleNode *node)
{
  auto tensor = std::make_shared<Tensor>();
  {
    tensor->dtype(node->dtype());
    tensor->rank(node->rank());
    for (uint32_t i = 0; i < node->rank(); i++)
      tensor->dim(i) = node->dim(i);

    switch (node->dtype())
    {
      case loco::DataType::FLOAT32:
        tensor->size<loco::DataType::FLOAT32>(numElements(node));
        break;
      case loco::DataType::U8:
        tensor->size<loco::DataType::U8>(numElements(node));
        break;
      case loco::DataType::S16:
        tensor->size<loco::DataType::S16>(numElements(node));
        break;
      case loco::DataType::S32:
        tensor->size<loco::DataType::S32>(numElements(node));
        break;
      case loco::DataType::S64:
        tensor->size<loco::DataType::S64>(numElements(node));
        break;
      default:
        throw std::runtime_error("Unsupported input tensor dtype for " + node->name());
    }
  }

  return tensor;
}

} // namespace

namespace circle_eval_diff
{

void H5InputEvalDiff::evalDiff(const std::string &first_input_data_path,
                               const std::string &second_input_data_path,
                               const std::string &output_prefix) const
{
  const auto first_interp = std::make_unique<luci_interpreter::Interpreter>(_first_module.get());
  const auto second_interp = std::make_unique<luci_interpreter::Interpreter>(_second_module.get());

  initMetrics(_first_module.get(), _second_module.get());

  try
  {
    HDF5Importer first_h5(first_input_data_path);
    first_h5.importGroup("value");

    HDF5Importer second_h5(second_input_data_path);
    second_h5.importGroup("value");

    const auto first_num_data = first_h5.numData();
    const auto second_num_data = second_h5.numData();

    if (first_num_data != second_num_data)
      throw std::runtime_error(
        "Number of data in the first data file and the second data file mismatches.");

    if (first_num_data == 0)
      throw std::runtime_error("Input data file does not contain any record.");

    const auto first_input_nodes = loco::input_nodes(_first_module->graph());
    const auto first_num_inputs = first_input_nodes.size();
    const auto first_output_nodes = loco::output_nodes(_first_module->graph());
    const auto first_num_outputs = first_output_nodes.size();

    const auto second_input_nodes = loco::input_nodes(_second_module->graph());
    const auto second_num_inputs = second_input_nodes.size();
    const auto second_output_nodes = loco::output_nodes(_second_module->graph());
    const auto second_num_outputs = second_output_nodes.size();

    for (int32_t data_idx = 0; data_idx < first_num_data; data_idx++)
    {
      std::cout << "Evaluating " << data_idx << "'th data" << std::endl;

      if (first_num_inputs != first_h5.numInputs(data_idx) ||
          second_num_inputs != second_h5.numInputs(data_idx))
        throw std::runtime_error("Wrong number of inputs in " + std::to_string(data_idx) +
                                 "th data.");

      // Do inference and return output
      auto eval = [&](luci_interpreter::Interpreter *interp, HDF5Importer &h5, uint32_t num_inputs,
                      const std::vector<loco::Node *> &input_nodes, uint32_t num_outputs,
                      const std::vector<loco::Node *> &output_nodes) {
        // Write input data
        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++)
        {
          const auto *input_node =
            loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
          assert(input_node->index() == input_idx);

          auto tensor = createEmptyTensor(input_node);
          if (h5.isRawData())
          {
            h5.readTensor(data_idx, input_idx, tensor->buffer());
          }
          else
          {
            DataType dtype;
            Shape shape;
            h5.readTensor(data_idx, input_idx, &dtype, &shape, tensor->buffer());

            // Check the type and the shape of the input data is valid
            verifyTypeShape(input_node, dtype, shape);
          }

          interp->writeInputTensor(input_node, tensor->buffer(), tensor->byte_size());
        }

        // Interpret
        interp->interpret();

        // Read output data
        std::vector<std::shared_ptr<Tensor>> outputs;
        for (uint32_t output_idx = 0; output_idx < num_outputs; output_idx++)
        {
          const auto *output_node =
            loco::must_cast<const luci::CircleOutput *>(output_nodes[output_idx]);
          assert(output_node->index() == output_idx);

          auto tensor = createEmptyTensor(output_node);
          interp->readOutputTensor(output_node, tensor->buffer(), tensor->byte_size());
          outputs.emplace_back(tensor);
        }

        return outputs;
      };

      auto first_output = eval(first_interp.get(), first_h5, first_num_inputs, first_input_nodes,
                               first_num_outputs, first_output_nodes);
      auto second_output = eval(second_interp.get(), second_h5, second_num_inputs,
                                second_input_nodes, second_num_outputs, second_output_nodes);

      // Accumulate diffs
      accumulateMetrics(first_output, second_output);

      // Save outputs
      for (uint32_t i = 0; i < first_output.size(); i++)
      {
        auto out = first_output[i];
        writeDataToFile(output_prefix + "." + std::to_string(data_idx) + ".first.output" +
                          std::to_string(i),
                        (char *)(out->buffer()), out->byte_size());
      }
      for (uint32_t i = 0; i < second_output.size(); i++)
      {
        auto out = second_output[i];
        writeDataToFile(output_prefix + "." + std::to_string(data_idx) + ".second.output" +
                          std::to_string(i),
                        (char *)(out->buffer()), out->byte_size());
      }
    }

    std::cout << "Evaluation finished. Number of data: " << first_num_data << std::endl;
  }
  catch (const H5::Exception &e)
  {
    H5::Exception::printErrorStack();
    throw std::runtime_error("HDF5 error occurred.");
  }

  // Print metric
  dumpMetrics();
}

} // namespace circle_eval_diff
