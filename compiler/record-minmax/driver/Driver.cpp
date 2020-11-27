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

  arser.add_argument("--input_model")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("Input model filepath");

  arser.add_argument("--input_data")
    .nargs(1)
    .type(arser::DataType::STR)
    .required(true)
    .help("Input data filepath");

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

  auto input_model_path = arser.get<std::string>("--input_model");
  auto input_data_path = arser.get<std::string>("--input_data");
  auto output_model_path = arser.get<std::string>("--output_model");

  // Default values
  std::string mode("percentile");
  float min_percentile = 1.0;
  float max_percentile = 99.0;

  if (arser["--min_percentile"])
    min_percentile = arser.get<float>("--min_percentile");

  if (arser["--max_percentile"])
    max_percentile = arser.get<float>("--max_percentile");

  if (arser["--mode"])
    mode = arser.get<std::string>("--mode");

  if (mode != "percentile" && mode != "moving_average")
    throw std::runtime_error("Unsupported mode");

  RecordMinMax rmm;

  // Initialize interpreter and observer
  rmm.initialize(input_model_path);

  // Profile min/max while executing the given input data
  rmm.profileData(mode, input_data_path, min_percentile, max_percentile);

  // Save profiled values to the model
  rmm.saveModel(output_model_path);

  return EXIT_SUCCESS;
}
