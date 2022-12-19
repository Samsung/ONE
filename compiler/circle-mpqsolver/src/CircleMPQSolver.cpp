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

#include "BisectionSolver.h"

#include <arser/arser.h>
#include <vconone/vconone.h>

#include <luci/ImporterEx.h>
#include <luci/Importer.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>

#include <iostream>
#include <iomanip>
#include <chrono>

void print_version(void)
{
  std::cout << "circle-mpqsolver version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int entry(int argc, char **argv)
{
  const std::string bisection_str = "--bisection";

  arser::Arser arser("circle-mpqsolver provides circle_model mixed precision quantization");


  arser::Helper::add_version(arser, print_version);
  arser::Helper::add_verbose(arser);

  arser.add_argument("--data").required(true).help(".h5 file with test data");

  arser.add_argument("--qerror_ratio")
    .type(arser::DataType::FLOAT)
    .default_value(0.5f)
    .help("quantization error ratio (> 0 and < 1)");

  arser.add_argument(bisection_str)
    .nargs(1)
    .default_value("auto")
    .type(arser::DataType::STR)
    .help("Bisection method Q16 at input nodes"
          "Default is 'auto'."
          "Single optional argument: whether Q16 qunatization should be at input nodes "
          "'auto', 'true', 'false'.");

  arser.add_argument("--input_model")
    .required(true)
    .help("Input float model with min max initialized");

  arser.add_argument("--output_model").required(true).help("Output quantized model");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cout << arser;
    return EXIT_FAILURE;
  }

  if (arser.get<bool>("--verbose"))
  {
    // The third parameter of setenv means REPLACE.
    // If REPLACE is zero, it does not overwrite an existing value.
    setenv("LUCI_LOG", "100", 0);
  }

  auto data_path = arser.get<std::string>("--data");
  auto input_model_path = arser.get<std::string>("--input_model");
  auto output_model_path = arser.get<std::string>("--output_model");

  float qerror_ratio = arser.get<float>("--qerror_ratio");
  if (qerror_ratio < 0.0 || qerror_ratio > 1.f)
  {
    std::cerr << "ERROR: quantization ratio should be above 0 and below 1" << std::endl;
    return EXIT_FAILURE;
  }
  auto start = std::chrono::high_resolution_clock::now();

  // Load input model
  luci::ImporterEx importerex;
  auto module = importerex.importVerifyModule(input_model_path);
  if (module.get() == nullptr)
  {
    std::cerr << "Failed to load " << input_model_path << std::endl;
    return EXIT_FAILURE;
  }

  // optimize
  mpqsolver::BisectionSolver solver(data_path, qerror_ratio);
  {
    auto value = arser.get<std::string>(bisection_str);
    if (value == "auto")
    {
      solver.options()->enable(mpqsolver::BisectionSolver::Options::Q16AtInput::Auto);
    }
    else if (value == "true")
    {
      solver.options()->enable(mpqsolver::BisectionSolver::Options::Q16AtInput::True);
    }
    else if (value == "false")
    {
      solver.options()->enable(mpqsolver::BisectionSolver::Options::Q16AtInput::False);
    }
    else
    {
      std::cerr << "Unrecognized option for bisection algortithm" << input_model_path << std::endl;
      return EXIT_FAILURE;
    }
  }

  auto optimized = solver.run(module.get());
  if (optimized == nullptr)
  {
    std::cerr << "Failed to build mixed precision model" << input_model_path << std::endl;
    return EXIT_FAILURE;
  }

  // save optimized
  {
    luci::CircleExporter exporter;
    luci::CircleFileExpContract contract(optimized.get(), output_model_path);
    if (!exporter.invoke(&contract))
    {
      std::cerr << "Failed to build mixed precision model" << input_model_path << std::endl;
      return EXIT_FAILURE;
    }
  }

  auto duration = std::chrono::duration_cast<std::chrono::seconds>(
    std::chrono::high_resolution_clock::now() - start);
  std::cerr << "Time elapsed is " << std::setprecision(5) << duration.count() / 60.f << " minutes."
            << std::endl;

  return EXIT_SUCCESS;
}
