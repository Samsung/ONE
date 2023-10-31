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

#include <arser/arser.h>
#include <vconone/vconone.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>

#include "bisection/BisectionSolver.h"
#include "pattern/PatternSolver.h"
#include "core/SolverOutput.h"

#include <iostream>
#include <iomanip>

void print_version(void)
{
  std::cout << "circle-mpqsolver version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int handleAutoAlgorithm(arser::Arser &arser, mpqsolver::bisection::BisectionSolver &solver)
{
  solver.algorithm(mpqsolver::bisection::BisectionSolver::Algorithm::Auto);
  auto data_path = arser.get<std::string>("--visq_file");
  if (data_path.empty())
  {
    std::cerr << "ERROR: please provide visq_file for auto mode" << std::endl;
    return false;
  }
  solver.setVisqPath(data_path);
  return true;
}

int entry(int argc, char **argv)
{
  const std::string bisection_str = "--bisection";
  const std::string patterns_str = "--patterns";
  const std::string layernorm_str = "--u8_layernorm_with_s16_variance";
  const std::string softmax_str = "--u8_softmax_with_s16_sub_exp";
  const std::string save_intermediate_str = "--save_intermediate";

  arser::Arser arser("circle-mpqsolver provides light-weight methods for finding a high-quality "
                     "mixed-precision model within a reasonable time.");

  arser::Helper::add_version(arser, print_version);
  arser::Helper::add_verbose(arser);

  // if patterns are set we don't need data
  arser.add_argument("--data").required(false).default_value("").help("Path to the test data");
  arser.add_argument("--data_format").required(false).help("Test data format (default: h5)");

  arser.add_argument("--qerror_ratio")
    .type(arser::DataType::FLOAT)
    .default_value(0.5f)
    .help("quantization error ratio ([0, 1])");

  arser.add_argument(bisection_str)
    .nargs(1)
    .required(false)
    .type(arser::DataType::STR)
    .help("Single optional argument for bisection method. "
          "Whether input node should be quantized to Q16: 'auto', 'true', 'false'.");

  arser.add_argument(patterns_str)
    .nargs(0)
    .required(false)
    .help("Argument to define patterns applied (LayerNorm is the only supported) ");

  arser.add_argument(layernorm_str)
    .nargs(0)
    .required(false)
    .help("Use int16 for computing variance in uint8 layer normalization");

  arser.add_argument(softmax_str)
    .nargs(0)
    .required(false)
    .help("Use int16 for computing sub and exp in uint8 softmax");

  arser.add_argument("--input_model")
    .required(true)
    .help("Input float model with min max initialized");

  arser.add_argument("--input_dtype")
    .type(arser::DataType::STR)
    .default_value("uint8")
    .help("Data type of quantized model's inputs (default: uint8)");

  arser.add_argument("--output_dtype")
    .type(arser::DataType::STR)
    .default_value("uint8")
    .help("Data type of quantized model's outputs (default: uint8)");

  arser.add_argument("--output_model").required(true).help("Output quantized model");

  arser.add_argument("--visq_file")
    .type(arser::DataType::STR)
    .default_value("")
    .required(false)
    .help("*.visq.json file with quantization errors");

  arser.add_argument(save_intermediate_str)
    .type(arser::DataType::STR)
    .required(false)
    .help("path to save intermediate results");

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
  auto input_dtype = arser.get<std::string>("--input_dtype");
  auto output_dtype = arser.get<std::string>("--output_dtype");

  float qerror_ratio = arser.get<float>("--qerror_ratio");
  if (qerror_ratio < 0.0 || qerror_ratio > 1.f)
  {
    std::cerr << "ERROR: quantization ratio must be in [0, 1]" << std::endl;
    return EXIT_FAILURE;
  }

  if (arser[bisection_str] && arser[patterns_str])
  {
    // only one solver can be used for now
    std::cerr << "ERROR: only one method is allowed to use" << std::endl;
    return EXIT_FAILURE;
  }

  SolverOutput::get() << ">> Searching mixed precision configuration \n"
                      << "model:" << input_model_path << "\n"
                      << "dataset: " << data_path << "\n"
                      << "input dtype: " << input_dtype << "\n"
                      << "output dtype: " << output_dtype << "\n";

  std::unique_ptr<mpqsolver::MPQSolver> solver;
  if (arser[bisection_str])
  {
    // optimize
    SolverOutput::get() << "using bisection\n";

    using namespace mpqsolver::bisection;
    auto bi_solver =
      std::make_unique<BisectionSolver>(data_path, qerror_ratio, input_dtype, output_dtype);

    {
      auto value = arser.get<std::string>(bisection_str);
      if (value == "auto")
      {
        SolverOutput::get() << "algorithm: bisection (auto)\n";
        if (!handleAutoAlgorithm(arser, *bi_solver))
        {
          return EXIT_FAILURE;
        }
      }
      else if (value == "true")
      {
        SolverOutput::get() << "algorithm: bisection (Q16AtFront)";
        bi_solver->algorithm(BisectionSolver::Algorithm::ForceQ16Front);
      }
      else if (value == "false")
      {
        SolverOutput::get() << "algorithm: bisection (Q8AtFront)";
        bi_solver->algorithm(BisectionSolver::Algorithm::ForceQ16Back);
      }
      else
      {
        std::cerr << "ERROR: Unrecognized option for bisection algortithm" << input_model_path
                  << std::endl;
        return EXIT_FAILURE;
      }
    }

    if (arser[save_intermediate_str])
    {
      auto data_path = arser.get<std::string>(save_intermediate_str);
      if (!data_path.empty())
      {
        bi_solver->set_save_intermediate(data_path);
      }
    }

    SolverOutput::get() << "qerror metric: MAE\n"
                        << "target qerror ratio: " << qerror_ratio << "\n";

    solver.reset(bi_solver.release());
  }
  else if (arser[patterns_str])
  {
    SolverOutput::get() << "using patterns\n";

    std::vector<mpqsolver::pattern::QuantizationPattern> patterns;
    if (arser[layernorm_str])
    {
      patterns.push_back(mpqsolver::pattern::QuantizationPattern::Q8LayerNormWithQ16Variance);
    }
    if (arser[softmax_str])
    {
      patterns.push_back(mpqsolver::pattern::QuantizationPattern::Q8SoftmaxWithQ16SubExp);
    }
    solver =
      std::make_unique<mpqsolver::pattern::PatternSolver>(input_dtype, output_dtype, patterns);
  }
  else
  {
    std::cerr << "ERROR: Unrecognized solver" << std::endl;
    return EXIT_FAILURE;
  }

  auto optimized = solver->run(input_model_path);
  if (optimized == nullptr)
  {
    std::cerr << "ERROR: Failed to build mixed precision model" << input_model_path << std::endl;
    return EXIT_FAILURE;
  }

  // save optimized
  {
    SolverOutput::get() << "Saving output model to " << output_model_path << "\n";
    luci::CircleExporter exporter;
    luci::CircleFileExpContract contract(optimized.get(), output_model_path);
    if (!exporter.invoke(&contract))
    {
      std::cerr << "ERROR: Failed to export mixed precision model" << input_model_path << std::endl;
      return EXIT_FAILURE;
    }
  }

  return EXIT_SUCCESS;
}
