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
#include "DataSetIterator.h"
#include "HDF5Iterator.h"
#include "RandomIterator.h"
#include "DirectoryIterator.h"
#include "ListFileIterator.h"
#include "Utils.h"

#include <luci/ImporterEx.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/Log.h>

#include <stdexcept>

using Shape = std::vector<loco::Dimension>;
using DataType = loco::DataType;

namespace
{

// Max h5 file size for parallel recording in bytes = 1 GB
const long h5_max_size_bytes = 1000000000;

long getH5FileSize(const std::string &input_data_path)
{
  std::ifstream in_file(input_data_path, std::ios::binary);
  in_file.seekg(0, std::ios::end);

  return in_file.tellg();
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

  luci::ImporterEx importer;
  _module = importer.importVerifyModule(input_model_path);

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

std::unique_ptr<DataSetIterator> RecordMinMax::createIterator()
{
  assert(_data_set_format != DataSetFormat::UNKNOWN); // FIX_CALLER_UNLESS

  std::unique_ptr<DataSetIterator> iterator;
  switch (_data_set_format)
  {
    case DataSetFormat::H5:
      assert(not _input_data_path.empty()); // FIX_CALLER_UNLESS
      iterator = std::make_unique<HDF5Iterator>(_input_data_path, _module.get());
      break;
    case DataSetFormat::RANDOM:
      iterator = std::make_unique<RandomIterator>(_module.get());
      break;
    case DataSetFormat::DIRECTORY:
      iterator = std::make_unique<DirectoryIterator>(_input_data_path, _module.get());
      break;
    case DataSetFormat::LIST_FILE:
      iterator = std::make_unique<ListFileIterator>(_input_data_path, _module.get());
      break;
    default:
      throw std::runtime_error("Unsupported dataset format");
  }

  assert(iterator.get() != nullptr); // FIX_ME_UNLESS

  return iterator;
}

void RecordMinMax::profileData()
{
  assert(getDataSetFormat() != DataSetFormat::UNKNOWN); // FIX_CALLER_UNLESS

  const auto input_nodes = loco::input_nodes(_module->graph());
  for (auto input_node : input_nodes)
  {
    const auto *input_cnode = loco::must_cast<const luci::CircleInput *>(input_node);
    checkInputDimension(input_cnode);
  }

  const auto num_inputs = input_nodes.size();

  auto iter = createIterator();

  bool check_type_shape = iter->check_type_shape();

  if (not iter->hasNext())
    throw std::runtime_error("The input data file does not contain any record.");

  uint32_t record_idx = 0;
  while (iter->hasNext())
  {
    const auto &record = iter->next();

    if (num_inputs != record.size())
      throw std::runtime_error("Wrong number of inputs.");

    std::cout << "Recording " << record_idx << "'th data" << std::endl;

    // Write input data to interpreter
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++)
    {
      const auto *input_node = loco::must_cast<const luci::CircleInput *>(input_nodes[input_idx]);
      assert(input_node->index() == input_idx);

      const auto input_data = record.at(input_idx);

      if (check_type_shape)
      {
        // Check the type and the shape of the input data is valid
        verifyTypeShape(input_node, input_data.dtype, input_data.shape);
      }

      getInterpreter()->writeInputTensor(input_node, input_data.data.data(),
                                         input_data.data.size());
    }

    getInterpreter()->interpret();

    record_idx++;
  }

  std::cout << "Recording finished. Number of recorded data: " << record_idx << std::endl;

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
