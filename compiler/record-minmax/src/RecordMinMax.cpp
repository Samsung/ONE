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
#include "RecordFunction.h"
#include "MinMaxObserver.h"
#include "HDF5Importer.h"

#include <luci/Importer.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/IR/CircleQuantParam.h>

#include <dirent.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <random>

using Shape = luci_interpreter::Shape;
using DataType = luci_interpreter::DataType;

namespace
{

void readDataFromFile(const std::string &filename, std::vector<char> &data, size_t data_size)
{
  assert(data.size() == data_size); // FIX_CALLER_UNLESS

  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.read(data.data(), data_size).fail())
    throw std::runtime_error("Failed to read data from file \"" + filename + "\".\n");
}

std::vector<uint8_t> genRandomBoolData(std::mt19937 &gen, uint32_t num_elements)
{
  std::uniform_int_distribution<> dist(0, 1);
  std::vector<uint8_t> input_data(num_elements);

  // Write random data
  for (auto &iter : input_data)
    iter = static_cast<uint8_t>(dist(gen));

  return input_data;
}

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
void verifyTypeShape(const luci::CircleInput *input_node, const DataType &dtype, const Shape &shape)
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

void update_quantparam(record_minmax::MinMaxObserver *observer, const std::string &mode,
                       float min_percentile, float max_percentile)
{
  auto minmax_map = observer->minMaxData()->getMap();
  for (auto iter = minmax_map->begin(); iter != minmax_map->end(); ++iter)
  {
    auto node = iter->first;
    auto minmax = iter->second;

    float min{0.0f}, max{0.0f};
    if (mode == "percentile")
    {
      min = record_minmax::getNthPercentile(minmax.min_vector, min_percentile);
      max = record_minmax::getNthPercentile(minmax.max_vector, max_percentile);
    }
    else if (mode == "moving_average")
    {
      min = record_minmax::getMovingAverage(minmax.min_vector, 0.9, 16, true);
      max = record_minmax::getMovingAverage(minmax.max_vector, 0.9, 16, false);
    }
    assert(mode == "percentile" || mode == "moving_average");
    auto quantparam = std::make_unique<luci::CircleQuantParam>();
    quantparam->min.push_back(min);
    quantparam->max.push_back(max);

    assert(node->quantparam() == nullptr);

    auto mutable_node = const_cast<luci::CircleNode *>(node);
    mutable_node->quantparam(std::move(quantparam));
  }
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

  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(model_data.data()),
                                 model_data.size()};
  if (!circle::VerifyModelBuffer(verifier))
  {
    throw std::runtime_error("Failed to verify circle '" + input_model_path + "'");
  }

  _module = luci::Importer().importModule(circle::GetModel(model_data.data()));

  if (_module == nullptr)
  {
    throw std::runtime_error("Failed to load '" + input_model_path + "'");
  }

  // Initialize interpreter
  _interpreter = std::make_unique<luci_interpreter::Interpreter>(_module.get());

  _observer = std::make_unique<MinMaxObserver>();

  _interpreter->attachObserver(_observer.get());
}

// input_data_path is a path to the directory
// The directory should contain binary files each of which is a raw data,
// ready to be consumed by the input circle model without any modification
// TODO reduce duplicate codes with profileRawData
void RecordMinMax::profileRawDataDirectory(const std::string &mode,
                                           const std::string &input_data_path, float min_percentile,
                                           float max_percentile)
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
    total_input_size += getTensorSize(input_node);
  }

  while (entry = readdir(dp))
  {
    // Skip if the entry is not a regular file
    if (entry->d_type != DT_REG)
      continue;

    const std::string filename = entry->d_name;
    std::cout << "Recording " << num_records << "'th data" << std::endl;

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
      _interpreter->writeInputTensor(input_node, input_data.data() + offset, input_size);

      offset += input_size;
    }

    _interpreter->interpret();

    num_records++;
  }

  closedir(dp);

  if (num_records == 0)
    throw std::runtime_error("The input data file does not contain any record.");

  std::cout << "Recording finished. Number of recorded data: " << num_records << std::endl;

  update_quantparam(_observer.get(), mode, min_percentile, max_percentile);
}

// input_data_path is a text file which specifies the representative data
// The text file should contain absolute file path per line.
// The pointed file should be a binary file containing one representative data,
// ready to be consumed by the input circle model without any modification
// NOTE If a model has multiple inputs, the binary file should have inputs concatenated in the same
// order with the input index of the circle model.
void RecordMinMax::profileRawData(const std::string &mode, const std::string &input_data_path,
                                  float min_percentile, float max_percentile)
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
    total_input_size += getTensorSize(input_node);
  }

  while (getline(input_file, record))
  {
    std::cout << "Recording " << num_records << "'th data" << std::endl;

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
      _interpreter->writeInputTensor(input_node, input_data.data() + offset, input_size);

      offset += input_size;
    }

    _interpreter->interpret();

    num_records++;
  }

  if (num_records == 0)
    throw std::runtime_error("The input data file does not contain any record.");

  std::cout << "Recording finished. Number of recorded data: " << num_records << std::endl;

  update_quantparam(_observer.get(), mode, min_percentile, max_percentile);
}

void RecordMinMax::profileData(const std::string &mode, const std::string &input_data_path,
                               float min_percentile, float max_percentile)
{
  try
  {
    HDF5Importer importer(input_data_path);
    importer.importGroup();

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

      std::cout << "Recording " << record_idx << "'th data" << std::endl;

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

        // TODO: Input data is copied twice (file -> buffer (input_data) -> interpreter inputs)
        //       We can redcue the copy by directly writing data from file to interpreter inputs
        _interpreter->writeInputTensor(input_node, input_data.data(), input_data.size());
      }

      _interpreter->interpret();
    }

    std::cout << "Recording finished. Number of recorded data: " << num_records << std::endl;
  }
  catch (const H5::Exception &e)
  {
    H5::Exception::printErrorStack();
    throw std::runtime_error("HDF5 error occurred.");
  }

  update_quantparam(_observer.get(), mode, min_percentile, max_percentile);
}

void RecordMinMax::profileDataWithRandomInputs(const std::string &mode, float min_percentile,
                                               float max_percentile)
{
  // We use three randomly-generated records
  const uint32_t num_records = 3;

  const auto input_nodes = loco::input_nodes(_module->graph());
  const auto num_inputs = input_nodes.size();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-5, 5);

  for (int32_t record_idx = 0; record_idx < num_records; record_idx++)
  {
    std::cout << "Recording " << record_idx << "'th data" << std::endl;

    for (int32_t input_idx = 0; input_idx < num_inputs; input_idx++)
    {
      const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
      assert(input_node->index() == input_idx);
      uint32_t num_elements = 1;
      for (uint32_t i = 0; i < input_node->rank(); i++)
      {
        if (!input_node->dim(i).known())
          throw std::runtime_error("Input dimension must be known");

        num_elements *= input_node->dim(i).value();
      }

      if (num_elements == 0)
        throw std::runtime_error("Only support non-zero sized inputs");

      // TODO Support more input data types
      assert(input_node->dtype() == loco::DataType::FLOAT32 ||
             input_node->dtype() == loco::DataType::BOOL);

      if (input_node->dtype() == DataType::FLOAT32)
      // clang-format off
      {
      std::vector<float> input_data(num_elements);

      // Write random data
      for (auto &iter : input_data)
        iter = static_cast<float>(dist(gen));

      // TODO: Input data is copied twice (file -> buffer (input_data) -> interpreter inputs)
      //       We can redcue the copy by directly writing data from file to interpreter inputs
      _interpreter->writeInputTensor(input_node, input_data.data(),
                                     input_data.size() * sizeof(float));
      }
      // clang-format on
      else if (input_node->dtype() == DataType::BOOL)
      {
        auto input_data = genRandomBoolData(gen, num_elements);
        _interpreter->writeInputTensor(input_node, input_data.data(),
                                       input_data.size() * sizeof(uint8_t));
      }
    }

    _interpreter->interpret();
  }

  std::cout << "Recording finished. Number of recorded data: " << num_records << std::endl;

  update_quantparam(_observer.get(), mode, min_percentile, max_percentile);
}

void RecordMinMax::saveModel(const std::string &output_model_path)
{
  // Export to output Circle file
  luci::CircleExporter exporter;

  luci::CircleFileExpContract contract(_module.get(), output_model_path);

  if (!exporter.invoke(&contract))
  {
    throw std::runtime_error("Failed to export '" + output_model_path + "'");
  }
}

} // namespace record_minmax
