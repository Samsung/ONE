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

void print_version(void)
{
  std::cout << "record-minmax version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int entry(const int argc, char **argv)
{
  using namespace record_minmax;

  arser::Arser arser(
    "Embedding min/max values of activations to the circle model for post-training quantization");

  arser.add_argument("--version")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Show version information and exit")
    .exit_with(print_version);

  arser.add_argument("-V", "--verbose")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("output additional information to stdout or stderr");

  arser.add_argument("--input_model")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("Input model filepath");

  arser.add_argument("--input_data")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(false)
    .help("Input data filepath. If not given, record-minmax will run with randomly generated data. "
          "Note that the random dataset does not represent inference workload, leading to poor "
          "model accuracy.");

  arser.add_argument("--output_model")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("Output model filepath");

  arser.add_argument("--min_percentile")
    .nargs(1)
    .type(arser::DataType::FLOAT)
    .help("Record n'th percentile of min");

  arser.add_argument("--max_percentile")
    .nargs(1)
    .type(arser::DataType::FLOAT)
    .help("Record n'th percentile of max");

  arser.add_argument("--mode")
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Record mode. percentile (default) or moving_average");

  arser.add_argument("--input_data_format")
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Input data format. h5/hdf5 (default) or list/filelist");

  arser.add_argument("--generate_profile_data")
    .nargs(0)
    .required(false)
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
    setenv("LUCI_LOG", "100", true);
  else
    setenv("LUCI_LOG", "0", true);

  auto settings = luci::UserSettings::settings();

  auto input_model_path = arser.get<std::string>("--input_model");
  auto output_model_path = arser.get<std::string>("--output_model");

  // Default values
  std::string mode("percentile");
  float min_percentile = 1.0;
  float max_percentile = 99.0;
  std::string input_data_format("h5");

  if (arser["--min_percentile"])
    min_percentile = arser.get<float>("--min_percentile");

  if (arser["--max_percentile"])
    max_percentile = arser.get<float>("--max_percentile");

  if (arser["--mode"])
    mode = arser.get<std::string>("--mode");

  if (mode != "percentile" && mode != "moving_average")
    throw std::runtime_error("Unsupported mode");

  if (arser["--generate_profile_data"])
    settings->set(luci::UserSettings::Key::ProfilingDataGen, true);

  if (arser["--input_data_format"])
    input_data_format = arser.get<std::string>("--input_data_format");

  RecordMinMax rmm;

  // Initialize interpreter and observer
  rmm.initialize(input_model_path);

  if (arser["--input_data"])
  {
    auto input_data_path = arser.get<std::string>("--input_data");

    if (input_data_format == "h5" || input_data_format == "hdf5")
    {
      // Profile min/max while executing the H5 data
      rmm.profileData(mode, input_data_path, min_percentile, max_percentile);
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
