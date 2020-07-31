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

#include "Dalgona.h"

#include <arser/arser.h>
#include <pybind11/embed.h>

namespace py = pybind11;

int entry(const int argc, char **argv)
{
  using namespace dalgona;

  arser::Arser arser("Dalgona: Dynamic analysis tool for DNN");

  arser.add_argument("--input_model")
      .nargs(1)
      .type(arser::DataType::STR)
      .required(true)
      .help("Input model filepath (.circle)");

  arser.add_argument("--input_data")
      .nargs(1)
      .type(arser::DataType::STR)
      .required(true)
      .help("Input data filepath (.h5)");

  arser.add_argument("--analysis")
      .nargs(1)
      .type(arser::DataType::STR)
      .required(true)
      .help("Analysis code filepath (.py)");

  arser.add_argument("--analysis_args")
      .nargs(1)
      .type(arser::DataType::STR)
      .help("String argument passed to the analysis code");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cout << err.what() << std::endl;
    std::cout << arser;
    return EXIT_FAILURE;
  }

  auto input_model_path = arser.get<std::string>("--input_model");
  auto input_data_path = arser.get<std::string>("--input_data");
  auto analysis_path = arser.get<std::string>("--analysis");
  std::string analysis_args = "";
  if (arser["--analysis_args"])
    analysis_args = arser.get<std::string>("--analysis_args");

  // Initialize python interpreter
  py::scoped_interpreter guard{};

  Dalgona dalgona;

  // Initialize interpreter and operator hooks
  dalgona.initialize(input_model_path);

  // Run analysis
  dalgona.runAnalysis(input_data_path, analysis_path, analysis_args);

  return EXIT_SUCCESS;
}
