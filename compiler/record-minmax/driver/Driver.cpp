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

#include <arser/arser.h>
#include <vconone/vconone.h>

#include <luci/UserSettings.h>

// TODO declare own log signature of record-minmax
#include <luci/Log.h>

void print_version(void)
{
  std::cout << "record-minmax version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int entry(const int argc, char **argv)
{
  using namespace record_minmax;

  LOGGER(l);

  arser::Arser arser(
    "Embedding min/max values of activations to the circle model for post-training quantization");

  arser::Helper::add_version(arser, print_version);
  arser::Helper::add_verbose(arser);

  arser.add_argument("--input_model").required(true).help("Input model filepath");

  arser.add_argument("--input_data")
    .help("Input data filepath. If not given, record-minmax will run with randomly generated data. "
          "Note that the random dataset does not represent inference workload, leading to poor "
          "model accuracy.");

  arser.add_argument("--output_model").required(true).help("Output model filepath");

  arser.add_argument("--min_percentile")
    .type(arser::DataType::FLOAT)
    .help("Record n'th percentile of min");

  arser.add_argument("--num_threads")
    .type(arser::DataType::INT32)
    .help("Number of threads (default: 1)");

  arser.add_argument("--max_percentile")
    .type(arser::DataType::FLOAT)
    .help("Record n'th percentile of max");

  arser.add_argument("--mode").help("Record mode. percentile (default) or moving_average");

  arser.add_argument("--moving_avg_batch")
    .type(arser::DataType::INT32)
    .help("Batch size of moving average algorithm (default: 16)");

  arser.add_argument("--moving_avg_const")
    .type(arser::DataType::FLOAT)
    .help("Hyperparameter (C) to compute moving average (default: 0.1). Update equation: avg <- "
          "avg + C * (curr_batch_avg - avg)");

  arser.add_argument("--input_data_format")
    .help("Input data format. h5/hdf5 (default) or list/filelist");

  arser.add_argument("--generate_profile_data")
    .nargs(0)
    .default_value(false)
    .help("This will turn on profiling data generation.");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << std::endl;
    std::cout << arser;
    return 255;
  }

  if (arser.get<bool>("--verbose"))
  {
    // The third parameter of setenv means REPLACE.
    // If REPLACE is zero, it does not overwrite an existing value.
    setenv("LUCI_LOG", "100", 0);
  }

  auto settings = luci::UserSettings::settings();

  auto input_model_path = arser.get<std::string>("--input_model");
  auto output_model_path = arser.get<std::string>("--output_model");

  // Default values
  std::string mode("percentile");
  float min_percentile = 1.0;
  float max_percentile = 99.0;
  uint32_t moving_avg_batch = 16;
  float moving_avg_const = 0.1;
  std::string input_data_format("h5");
  uint32_t num_threads = 1;

  if (arser["--min_percentile"])
    min_percentile = arser.get<float>("--min_percentile");

  if (arser["--num_threads"])
    num_threads = arser.get<int>("--num_threads");

  if (num_threads < 1)
    throw std::runtime_error("The number of threads must be greater than zero");

  if (arser["--max_percentile"])
    max_percentile = arser.get<float>("--max_percentile");

  if (arser["--mode"])
    mode = arser.get<std::string>("--mode");

  if (arser["--moving_avg_batch"])
    moving_avg_batch = arser.get<int>("--moving_avg_batch");

  if (arser["--moving_avg_const"])
    moving_avg_const = arser.get<float>("--moving_avg_const");

  if (mode != "percentile" && mode != "moving_average")
    throw std::runtime_error("Unsupported mode");

  if (arser["--generate_profile_data"])
    settings->set(luci::UserSettings::Key::ProfilingDataGen, true);

  if (arser["--input_data_format"])
    input_data_format = arser.get<std::string>("--input_data_format");

  std::unique_ptr<MinMaxComputer> computer;
  {
    if (mode == "percentile")
    {
      computer = make_percentile_computer(min_percentile, max_percentile);
    }
    else if (mode == "moving_average")
    {
      computer = make_moving_avg_computer(moving_avg_batch, moving_avg_const);
    }
    else
    {
      assert(false);
    }
  }

  RecordMinMax rmm(num_threads, std::move(computer));

  // TODO: support parallel record for profile with random data
  if (num_threads > 1 and not arser["--input_data"])
  {
    throw std::runtime_error("Input data must be given for parallel recording");
  }

  // Initialize interpreter and observer
  rmm.initialize(input_model_path);

  if (arser["--input_data"])
  {
    auto input_data_path = arser.get<std::string>("--input_data");

    // TODO: support parallel record from file and dir input data format
    if (num_threads > 1 and not(input_data_format == "h5") and not(input_data_format == "hdf5"))
    {
      throw std::runtime_error("Parallel recording is used only for h5 now");
    }

    if (input_data_format == "h5" || input_data_format == "hdf5")
    {
      // Profile min/max while executing the H5 data
      if (num_threads == 1)
        rmm.profileData(mode, input_data_path, min_percentile, max_percentile);
      else
      {
        INFO(l) << "Using parallel recording" << std::endl;
        rmm.profileDataInParallel(mode, input_data_path, min_percentile, max_percentile);
      }
    }
    // input_data is a text file having a file path in each line.
    // Each data file is composed of inputs of a model, concatenated in
    // the same order with the input index of the model
    //
    // For example, for a model with n inputs, the contents of each data
    // file can be visualized as below
    // [input 1][input 2]...[input n]
    // |start............end of file|
    else if (input_data_format == "list" || input_data_format == "filelist")
    {
      // Profile min/max while executing the list of Raw data
      rmm.profileRawData(mode, input_data_path, min_percentile, max_percentile);
    }
    else if (input_data_format == "directory" || input_data_format == "dir")
    {
      // Profile min/max while executing all files under the given directory
      // The contents of each file is same as the raw data in the 'list' type
      rmm.profileRawDataDirectory(mode, input_data_path, min_percentile, max_percentile);
    }
    else
    {
      throw std::runtime_error(
        "Unsupported input data format (supported formats: h5/hdf5 (default), list/filelist)");
    }
  }
  else
  {
    // Profile min/max while executing random input data
    rmm.profileDataWithRandomInputs(mode, min_percentile, max_percentile);
  }

  // Save profiled values to the model
  rmm.saveModel(output_model_path);

  return EXIT_SUCCESS;
}
