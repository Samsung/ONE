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

#include "Dalgona.h"
#include "PythonHooks.h"
#include "RandomUtils.h"

#include <luci/IR/DataTypeHelper.h>
#include <luci/Importer.h>
#include <foder/FileLoader.h>
#include <dio_hdf5/HDF5Importer.h>

#include <pybind11/embed.h>

#include <iostream>
#include <limits>

using Shape = std::vector<loco::Dimension>;
using DataType = loco::DataType;

namespace py = pybind11;

namespace
{

uint32_t numElements(const luci::CircleNode *node)
{
  assert(node != nullptr); // FIX_CALLER_UNLESS

  uint32_t num_elements = 1;
  for (uint32_t i = 0; i < node->rank(); i++)
    num_elements *= node->dim(i).value();

  return num_elements;
}

// Return tensor's size in bytes
template <typename NodeT> size_t getByteSize(const NodeT *node)
{
  assert(node != nullptr); // FIX_CALLER_UNLESS

  uint32_t dtype_size = luci::size(node->dtype());
  return static_cast<size_t>(dtype_size) * static_cast<size_t>(numElements(node));
}

// Throw exception if input has one of the following conditions.
// 1. Have unknown dimension
// 2. Number of elements is 0
void checkInputDimension(const luci::CircleInput *input)
{
  assert(input != nullptr); // FIX_CALLER_UNLESS

  for (uint32_t i = 0; i < input->rank(); i++)
    if (!input->dim(i).known())
      throw std::runtime_error(input->name() + " has unknown dimension");

  if (numElements(input) == 0)
    throw std::runtime_error(input->name() + " is a zero-sized input");
}

// Check the type and the shape of CircleInput
// Throw an exception if type or shape does not match
void verifyTypeShape(const luci::CircleInput *input_node, const DataType &dtype, const Shape &shape)
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

namespace dalgona
{

void Dalgona::initialize(const std::string &input_model_path)
{
  // Load model from the file
  foder::FileLoader loader{input_model_path};
  std::vector<char> model_data = loader.load();

  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(model_data.data()),
                                 model_data.size()};
  if (not circle::VerifyModelBuffer(verifier))
    throw std::runtime_error("Failed to verify circle '" + input_model_path + "'");

  auto circle_model = circle::GetModel(model_data.data());

  if (not circle_model)
    throw std::runtime_error("Failed to load '" + input_model_path + "'");

  _module = luci::Importer().importModule(circle_model);

  if (not _module)
    throw std::runtime_error("ERROR: Failed to load '" + input_model_path + "'");

  // Initialize interpreter
  _interpreter = std::make_unique<luci_interpreter::Interpreter>(_module.get());

  _hooks = std::make_unique<PythonHooks>(_interpreter.get());

  _interpreter->attachObserver(_hooks.get());
}

void Dalgona::runAnalysisWithH5Input(const std::string &input_data_path,
                                     const std::string &analysis_path,
                                     const std::string &analysis_args)
{
  py::object scope = py::module::import("__main__").attr("__dict__");
  _hooks->importAnalysis(analysis_path, scope, analysis_args);

  try
  {
    dio::hdf5::HDF5Importer importer(input_data_path);
    importer.importGroup("value");

    bool is_raw_data = importer.isRawData();

    const auto num_records = importer.numData();
    if (num_records == 0)
      throw std::runtime_error("The input data file does not contain any record.");

    const auto input_nodes = loco::input_nodes(_module->graph());
    const auto num_inputs = input_nodes.size();

    for (int32_t record_idx = 0; record_idx < num_records; record_idx++)
    {
      if (num_inputs != static_cast<uint32_t>(importer.numInputs(record_idx)))
        throw std::runtime_error("Wrong number of inputs.");

      std::cout << "Running " << record_idx << "'th data" << std::endl;

      for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++)
      {
        const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
        assert(input_node->index() == input_idx);
        checkInputDimension(input_node);
        std::vector<char> input_data(getByteSize(input_node));

        if (is_raw_data)
        {
          // Skip type/shape check for raw data
          importer.readTensor(record_idx, input_idx, input_data.data(), input_data.size());
        }
        else
        {
          DataType dtype;
          Shape shape;
          importer.readTensor(record_idx, input_idx, &dtype, &shape, input_data.data(),
                              input_data.size());

          // Check the type and the shape of the input data is valid
          verifyTypeShape(input_node, dtype, shape);
        }

        _interpreter->writeInputTensor(input_node, input_data.data(), input_data.size());
      }

      _hooks->startNetworkExecution(_module->graph());
      _interpreter->interpret();
      _hooks->endNetworkExecution(_module->graph());
    }

    std::cout << "Finished executing " << num_records << "'th data" << std::endl;
    _hooks->endAnalysis();
  }
  catch (const H5::Exception &e)
  {
    H5::Exception::printErrorStack();
    throw std::runtime_error("HDF5 error occurred.");
  }
}

void Dalgona::runAnalysisWithRandomInput(const std::string &analysis_path,
                                         const std::string &analysis_args)
{
  py::object scope = py::module::import("__main__").attr("__dict__");
  _hooks->importAnalysis(analysis_path, scope, analysis_args);

  const auto input_nodes = loco::input_nodes(_module->graph());
  const auto num_inputs = input_nodes.size();

  for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++)
  {
    const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
    assert(input_node->index() == input_idx);
    checkInputDimension(input_node);

    uint32_t num_elems = numElements(input_node);
    switch (input_node->dtype())
    {
      case DataType::FLOAT32:
      {
        // Synced with record-minmax (-5,5)
        auto input_data = genRandomFloatData(num_elems, -5, 5);
        _interpreter->writeInputTensor(input_node, input_data.data(),
                                       input_data.size() * sizeof(float));
        break;
      }
      case DataType::U8:
      {
        auto input_data = genRandomIntData<uint8_t>(num_elems, std::numeric_limits<uint8_t>::min(),
                                                    std::numeric_limits<uint8_t>::max());
        _interpreter->writeInputTensor(input_node, input_data.data(),
                                       input_data.size() * sizeof(uint8_t));
        break;
      }
      case DataType::S16:
      {
        auto input_data = genRandomIntData<int16_t>(num_elems, std::numeric_limits<int16_t>::min(),
                                                    std::numeric_limits<int16_t>::max());
        _interpreter->writeInputTensor(input_node, input_data.data(),
                                       input_data.size() * sizeof(int16_t));
        break;
      }
      case DataType::S32:
      {
        // Synced with record-minmax (0, 100)
        auto input_data = genRandomIntData<int32_t>(num_elems, 0, 100);
        _interpreter->writeInputTensor(input_node, input_data.data(),
                                       input_data.size() * sizeof(int32_t));
        break;
      }
      case DataType::S64:
      {
        // Synced with record-minmax (0, 100)
        auto input_data = genRandomIntData<int64_t>(num_elems, 0, 100);
        _interpreter->writeInputTensor(input_node, input_data.data(),
                                       input_data.size() * sizeof(int64_t));
        break;
      }
      case DataType::BOOL:
      {
        // Bool is represented as uint8 (0 or 1)
        auto input_data = genRandomIntData<uint8_t>(num_elems, 0, 1);
        _interpreter->writeInputTensor(input_node, input_data.data(),
                                       input_data.size() * sizeof(uint8_t));
        break;
      }
      default:
        throw std::runtime_error("Unsupported input data type in " + input_node->name());
    }
  }

  _hooks->startNetworkExecution(_module->graph());
  _interpreter->interpret();
  _hooks->endNetworkExecution(_module->graph());

  std::cout << "Finished executing a random input" << std::endl;
  _hooks->endAnalysis();
}

} // namespace dalgona
