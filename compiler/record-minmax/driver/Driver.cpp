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

int entry(const int argc, char **argv)
{
  using namespace record_minmax;

  arser::Arser arser(
      "Embedding min/max values of activations to the circle model for post-training quantization");

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

  arser.parse(argc, argv);

  auto input_model_path = arser.get<std::string>("--input_model");
  auto input_data_path = arser.get<std::string>("--input_data");
  auto output_model_path = arser.get<std::string>("--output_model");

  RecordMinMax rmm;

  // Initialize interpreter and observer
  rmm.initialize(input_model_path);

  // Profile min/max while executing the given input data
  rmm.profileData(input_data_path);

  // Save profiled values to the model
  rmm.saveModel(output_model_path);

  return EXIT_SUCCESS;
}
