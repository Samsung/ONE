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
#include "MinMaxObserver.h"

#include <luci/Importer.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
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

// Max h5 file size for parallel recording in bytes = 1 GB
const long h5_max_size_bytes = 1000000000;

long getH5FileSize(const std::string &input_data_path)
{
  std::ifstream in_file(input_data_path, std::ios::binary);
  in_file.seekg(0, std::ios::end);

  return in_file.tellg();
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

std::vector<uint8_t> genRandomBoolData(std::mt19937 &gen, uint32_t num_elements)
{
  std::uniform_int_distribution<> dist(0, 1);
  std::vector<uint8_t> input_data(num_elements);

  // Write random data
  for (auto &iter : input_data)
    iter = static_cast<uint8_t>(dist(gen));

  return input_data;
}

template <typename T>
std::vector<T> genRandomIntData(std::mt19937 &gen, uint32_t num_elements, T min, T max)
{
  std::uniform_int_distribution<T> dist(min, max);
  std::vector<T> input_data(num_elements);

  // Write random data
  {
    auto const generator = [&gen, &dist]() { return dist(gen); };
    std::generate(begin(input_data), end(input_data), generator);
  }

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

  if (shape.size() != input_node->rank())
    throw std::runtime_error("Input rank mismatch.");

  for (uint32_t i = 0; i < shape.size(); i++)
  {
    if (not(shape.at(i) == input_node->dim(i)))
      throw std::runtime_error("Input shape mismatch.");
  }
}

} // namespace

namespace record_minmax
{

void RecordMinMax::initialize(const std::string &input_model_path)
{
  assert(_threads_size > 0);

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

  const circle::Model *circle_model = circle::GetModel(model_data.data());
  if (circle_model == nullptr)
  {
    throw std::runtime_error("Failed to load '" + input_model_path + "'");
  }

  _module = luci::Importer().importModule(circle_model);

  if (_module == nullptr)
  {
    throw std::runtime_error("Failed to load '" + input_model_path + "'");
  }

  // Create and initialize interpreters and observers
  _interpreters.resize(_threads_size);
  _observers.resize(_threads_size);

  for (uint32_t thread_idx = 0; thread_idx < _threads_size; ++thread_idx)
  {
    auto interpreter = std::make_unique<luci_interpreter::Interpreter>(_module.get());
    auto observer = std::make_unique<MinMaxObserver>();

    interpreter->attachObserver(observer.get());

    _observers[thread_idx] = std::move(observer);
    _interpreters[thread_idx] = std::move(interpreter);
  }
}

// input_data_path is a path to the directory
// The directory should contain binary files each of which is a raw data,
// ready to be consumed by the input circle model without any modification
// TODO reduce duplicate codes with profileRawData
void RecordMinMax::profileRawDataDirectory(const std::string &input_data_path)
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

  _minmax_computer->update_qparam(getObserver()->minMaxData()->getMap());
}

// input_data_path is a text file which specifies the representative data
// The text file should contain absolute file path per line.
// The pointed file should be a binary file containing one representative data,
// ready to be consumed by the input circle model without any modification
// NOTE If a model has multiple inputs, the binary file should have inputs concatenated in the same
// order with the input index of the circle model.
void RecordMinMax::profileRawData(const std::string &input_data_path)
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
    std::cout << "Recording " << num_records << "'th data" << std::endl;

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

  _minmax_computer->update_qparam(getObserver()->minMaxData()->getMap());
}

WholeOutput RecordMinMax::importH5Data(const std::string &input_data_path)
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

    WholeOutput whole_output(num_records);

    // Read inputs to whole_output
    for (int i = 0; i < num_records; ++i)
    {
      if (num_inputs != static_cast<uint32_t>(importer.numInputs(i)))
        throw std::runtime_error("Wrong number of inputs.");

      for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++)
      {
        const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
        assert(input_node->index() == input_idx);
        checkInputDimension(input_node);
        Buffer input_data(getTensorSize(input_node));

        if (!is_raw_data)
        {
          DataType dtype;
          Shape shape;
          importer.readTensor(i, input_idx, &dtype, &shape, input_data.data(), input_data.size());

          // Check the type and the shape of the input data is valid
          verifyTypeShape(input_node, dtype, shape);
        }
        else
        {
          // Skip type/shape check for raw data
          importer.readTensor(i, input_idx, input_data.data(), input_data.size());
        }
        whole_output[i].emplace_back(std::move(input_data));
      }
    }

    return whole_output;
  }
  catch (const H5::Exception &e)
  {
    H5::Exception::printErrorStack();
    throw std::runtime_error("HDF5 error occurred.");
  }
}

void RecordMinMax::profileData(const std::string &input_data_path)
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

      std::cout << "Recording " << record_idx << "'th data" << std::endl;

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

  _minmax_computer->update_qparam(getObserver()->minMaxData()->getMap());
}

void RecordMinMax::profileDataInParallel(const std::string &input_data_path)
{
  LOGGER(l);

  assert(_interpreters.size() == _threads_size);
  assert(_observers.size() == _threads_size);

  const long h5_file_size = getH5FileSize(input_data_path);

  if (h5_file_size > h5_max_size_bytes)
    throw std::runtime_error("H5 file size is too large for parallel recording");

  WholeOutput whole_output;
  try
  {
    whole_output = importH5Data(input_data_path);
  }
  catch (const std::bad_alloc &e)
  {
    throw std::runtime_error("Out of memory during h5 data load.");
  }

  const auto num_records = whole_output.size();
  const auto input_nodes = loco::input_nodes(_module->graph());

  // Start parallel part
  INFO(l) << _threads_size << " concurrent threads are supported." << std::endl;

  const auto run_threads = num_records < _threads_size ? num_records : _threads_size;

  const auto records_batch = static_cast<uint32_t>(num_records / run_threads);

  auto interpret_batch = [&whole_output, &input_nodes](int first_record, int last_record,
                                                       luci_interpreter::Interpreter *interpreter) {
    for (int record_index = first_record; record_index < last_record; ++record_index)
    {
      for (uint32_t input_idx = 0; input_idx < input_nodes.size(); input_idx++)
      {
        const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);

        const auto &cur_input_data = whole_output[record_index][input_idx];
        interpreter->writeInputTensor(input_node, cur_input_data.data(), cur_input_data.size());
      }
      interpreter->interpret();
    }
  };

  std::vector<std::thread> threads;
  for (uint32_t t = 0; t < run_threads; ++t)
  {
    if (t < run_threads - 1)
    {
      threads.emplace_back(interpret_batch, records_batch * t, records_batch * (t + 1),
                           _interpreters[t].get());
    }
    else
    {
      threads.emplace_back(interpret_batch, records_batch * t, num_records, _interpreters[t].get());
    }
  }

  for (uint32_t i = 0; i < run_threads; ++i)
    threads.at(i).join();

  // End parallel part

  // Copy all min, max values to one min/max map
  MinMaxMap main_min_max_map;

  for (const auto &obs : _observers)
  {
    const auto cur_minmax_map = obs->minMaxData()->getMap();
    for (auto &iter : *cur_minmax_map)
    {
      const auto node = iter.first;
      const auto &minmax = iter.second;

      main_min_max_map.appendMinMaxVector(node, minmax);
    }
  }

  std::cout << "Recording finished. Number of recorded data: " << num_records << std::endl;

  _minmax_computer->update_qparam(main_min_max_map.getMap());
}

void RecordMinMax::profileDataWithRandomInputs(void)
{
  // We use three randomly-generated records
  const uint32_t num_records = 3;

  const auto input_nodes = loco::input_nodes(_module->graph());
  const auto num_inputs = input_nodes.size();

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(-5, 5);

  for (uint32_t record_idx = 0; record_idx < num_records; record_idx++)
  {
    std::cout << "Recording " << record_idx << "'th data" << std::endl;

    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++)
    {
      const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
      assert(input_node->index() == input_idx);
      checkInputDimension(input_node);

      const auto num_elements = numElements(input_node);

      // TODO Support more input data types
      assert(input_node->dtype() == loco::DataType::FLOAT32 ||
             input_node->dtype() == loco::DataType::BOOL ||
             input_node->dtype() == loco::DataType::S32 ||
             input_node->dtype() == loco::DataType::S64);

      if (input_node->dtype() == DataType::FLOAT32)
      {
        std::vector<float> input_data(num_elements);

        // Write random data
        for (auto &iter : input_data)
          iter = static_cast<float>(dist(gen));

        // TODO: Input data is copied twice (file -> buffer (input_data) -> interpreter inputs)
        //       We can redcue the copy by directly writing data from file to interpreter inputs
        getInterpreter()->writeInputTensor(input_node, input_data.data(),
                                           input_data.size() * sizeof(float));
      }
      else if (input_node->dtype() == DataType::BOOL)
      {
        auto input_data = genRandomBoolData(gen, num_elements);
        getInterpreter()->writeInputTensor(input_node, input_data.data(),
                                           input_data.size() * sizeof(uint8_t));
      }
      else if (input_node->dtype() == DataType::S32)
      {
        auto input_data = genRandomIntData<int32_t>(gen, num_elements, 0, 100);
        getInterpreter()->writeInputTensor(input_node, input_data.data(),
                                           input_data.size() * sizeof(int32_t));
      }
      else if (input_node->dtype() == DataType::S64)
      {
        auto input_data = genRandomIntData<int64_t>(gen, num_elements, 0, 100);
        getInterpreter()->writeInputTensor(input_node, input_data.data(),
                                           input_data.size() * sizeof(int64_t));
      }
    }

    getInterpreter()->interpret();
  }

  std::cout << "Recording finished. Number of recorded data: " << num_records << std::endl;

  _minmax_computer->update_qparam(getObserver()->minMaxData()->getMap());
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
