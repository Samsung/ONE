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

#include "record-hessian/RecordHessian.h"
#include "record-hessian/HessianObserver.h"

#include <luci/IR/DataTypeHelper.h>
#include <luci/Importer.h>
#include <luci/IR/CircleQuantParam.h>
#include <luci/Log.h>
#include <dio_hdf5/HDF5Importer.h>

#include <dirent.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <random>

using Shape = std::vector<loco::Dimension>;
using DataType = loco::DataType;

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

uint32_t numElements(const luci::CircleNode *node)
{
  uint32_t num_elements = 1;
  for (uint32_t i = 0; i < node->rank(); i++)
    num_elements *= node->dim(i).value();

  return num_elements;
}

// Throw exception if input has one of the following conditions.
// 1. Have unknown dimension
// 2. Number of elements is 0
void checkInputDimension(const luci::CircleInput *input)
{
  for (uint32_t i = 0; i < input->rank(); i++)
    if (!input->dim(i).known())
      throw std::runtime_error(input->name() + " has unknown dimension");

  if (numElements(input) == 0)
    throw std::runtime_error(input->name() + " is a zero-sized input");
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

/**
 * @brief  getTensorSize will return size in bytes
 */
template <typename NodeT> size_t getTensorSize(const NodeT *node)
{
  uint32_t tensor_size = luci::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

/**
 * @brief  verifyTypeShape checks the type and the shape of CircleInput
 *         This throws an exception if type or shape does not match
 */
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

} // namespace

namespace record_hessian
{

void RecordHessian::initialize(luci::Module *module)
{

  // Create and initialize interpreters and observers

  _module = module;

  auto interpreter = std::make_unique<luci_interpreter::Interpreter>(module);
  auto observer = std::make_unique<HessianObserver>();

  interpreter->attachObserver(observer.get());

  _observer = std::move(observer);
  _interpreter = std::move(interpreter);
}

// input_data_path is a path to the directory
// The directory should contain binary files each of which is a raw data,
// ready to be consumed by the input circle model without any modification
// TODO reduce duplicate codes with profileRawData
std::unique_ptr<HessianMap>
RecordHessian::profileRawDataDirectory(const std::string &input_data_path)
{
  struct dirent *entry = nullptr;
  DIR *dp = nullptr;

  dp = opendir(input_data_path.c_str());
  if (not dp)
    throw std::runtime_error("Cannot open directory. Please check \"" + input_data_path +
                             "\" is a directory.\n");

  uint32_t num_records = 0;
  const auto input_nodes = loco::input_nodes(_module->graph());

  // Get total input size
  uint32_t total_input_size = 0;
  for (auto input : input_nodes)
  {
    const auto *input_node = loco::must_cast<const luci::CircleInput *>(input);
    checkInputDimension(input_node);
    total_input_size += getTensorSize(input_node);
  }

  while ((entry = readdir(dp)))
  {
    // Skip if the entry is not a regular file
    if (entry->d_type != DT_REG)
      continue;

    const std::string filename = entry->d_name;
    std::cout << "Recording " << num_records << "'th data for hessian" << std::endl;

    // Read data from file to buffer
    // Assumption: For a multi-input model, the binary file should have inputs concatenated in the
    // same order with the input index.
    std::vector<char> input_data(total_input_size);
    readDataFromFile(input_data_path + "/" + filename, input_data, total_input_size);

    // Write data from buffer to interpreter
    uint32_t offset = 0;
    for (auto input : input_nodes)
    {
      const auto *input_node = loco::must_cast<const luci::CircleInput *>(input);
      const auto input_size = getTensorSize(input_node);
      getInterpreter()->writeInputTensor(input_node, input_data.data() + offset, input_size);

      offset += input_size;
    }

    getInterpreter()->interpret();

    num_records++;
  }

  closedir(dp);

  if (num_records == 0)
    throw std::runtime_error("The input data file does not contain any record.");

  std::cout << "Recording finished. Number of recorded data: " << num_records << std::endl;

  return getObserver()->hessianData();
}

// input_data_path is a text file which specifies the representative data
// The text file should contain absolute file path per line.
// The pointed file should be a binary file containing one representative data,
// ready to be consumed by the input circle model without any modification
// NOTE If a model has multiple inputs, the binary file should have inputs concatenated in the
// same order with the input index of the circle model.
std::unique_ptr<HessianMap> RecordHessian::profileRawData(const std::string &input_data_path)
{
  std::ifstream input_file(input_data_path);
  if (input_file.fail())
    throw std::runtime_error("Cannot open file \"" + input_data_path + "\".\n");

  std::string record;
  uint32_t num_records = 0;
  const auto input_nodes = loco::input_nodes(_module->graph());

  // Get total input size
  uint32_t total_input_size = 0;
  for (auto input : input_nodes)
  {
    const auto *input_node = loco::must_cast<const luci::CircleInput *>(input);
    checkInputDimension(input_node);
    total_input_size += getTensorSize(input_node);
  }

  while (getline(input_file, record))
  {
    std::cout << "Recording " << num_records << "'th data for hessian" << std::endl;

    auto file_names = parse_line(record);

    // Have multiple files in one line
    if (file_names.size() == input_nodes.size())
    {
      std::vector<std::vector<char>> input_data;
      for (uint32_t i = 0; i < file_names.size(); i++)
      {
        const auto file_name = file_names[i];
        const auto input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[i]);
        const auto input_size = getTensorSize(input_node);

        input_data.emplace_back(input_size);

        // Read data from file
        readDataFromFile(file_name, input_data[i], input_size);

        // Write data from buffer to interpreter
        getInterpreter()->writeInputTensor(input_node, input_data[i].data(), input_size);
      }

      getInterpreter()->interpret();

      num_records++;
    }
    else
    {
      // Must have a single file in one line (inputs are concatenated)
      if (file_names.size() != 1)
        throw std::runtime_error(
          "Wrong number of inputs are given. Model has " + std::to_string(input_nodes.size()) +
          " inputs, but list file gives " + std::to_string(file_names.size()) + " inputs.");

      // clang-format off
    // Read data from file to buffer
    // Assumption: For a multi-input model, the binary file should have inputs concatenated in the
    // same order with the input index.
    std::vector<char> input_data(total_input_size);
    readDataFromFile(record, input_data, total_input_size);

    // Write data from buffer to interpreter
    uint32_t offset = 0;
    for (auto input : input_nodes)
    {
      const auto *input_node = loco::must_cast<const luci::CircleInput *>(input);
      const auto input_size = getTensorSize(input_node);
      getInterpreter()->writeInputTensor(input_node, input_data.data() + offset, input_size);

      offset += input_size;
    }

    getInterpreter()->interpret();

    num_records++;
      // clang-format on
    }
  }

  if (num_records == 0)
    throw std::runtime_error("The input data file does not contain any record.");

  std::cout << "Recording finished. Number of recorded data: " << num_records << std::endl;

  return getObserver()->hessianData();
}

std::unique_ptr<HessianMap> RecordHessian::profileData(const std::string &input_data_path)
{
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

      std::cout << "Recording " << record_idx << "'th data for hessian" << std::endl;

      for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++)
      {
        const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
        assert(input_node->index() == input_idx);
        checkInputDimension(input_node);
        std::vector<char> input_data(getTensorSize(input_node));

        if (!is_raw_data)
        {
          DataType dtype;
          Shape shape;
          importer.readTensor(record_idx, input_idx, &dtype, &shape, input_data.data(),
                              input_data.size());

          // Check the type and the shape of the input data is valid
          verifyTypeShape(input_node, dtype, shape);
        }
        else
        {
          // Skip type/shape check for raw data
          importer.readTensor(record_idx, input_idx, input_data.data(), input_data.size());
        }

        // TODO: Input data is copied twice (file -> buffer (input_data) -> interpreter inputs)
        //       We can redcue the copy by directly writing data from file to interpreter inputs
        getInterpreter()->writeInputTensor(input_node, input_data.data(), input_data.size());
      }

      getInterpreter()->interpret();
    }

    std::cout << "Recording finished. Number of recorded data: " << num_records << std::endl;
  }
  catch (const H5::Exception &e)
  {
    H5::Exception::printErrorStack();
    throw std::runtime_error("HDF5 error occurred.");
  }

  return getObserver()->hessianData();
}

} // namespace record_hessian
