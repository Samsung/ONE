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

#include "CircleExpContract.h"

#include <foder/FileLoader.h>

#include <luci/Importer.h>
#include <luci/CircleOptimizer.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>

#include <oops/InternalExn.h>

#include <functional>
#include <iostream>
#include <map>
#include <string>

using OptionHook = std::function<int(const char **)>;

using Algorithms = luci::CircleOptimizer::Options::Algorithm;
using AlgorithmParameters = luci::CircleOptimizer::Options::AlgorithmParameters;

void print_help(const char *progname)
{
  std::cerr << "USAGE: " << progname << " [options] input output" << std::endl;
  std::cerr << "Options: " << std::endl;
  std::cerr << "   --quantize_with_minmax : Enable QuantizeWithMinMax Pass" << std::endl;
  std::cerr << "                            ";
  std::cerr << "Require three following parameters (input_dtype, quantized_dtype, granularity)"
            << std::endl;
  std::cerr << "                            ";
  std::cerr << "Ex: --quantize_with_minmax float32 uint8 layer" << std::endl;
  std::cerr << "   --quantize_dequantize_weights : Enable QuantizeDequantizeWeights Pass"
            << std::endl;
  std::cerr << "                            ";
  std::cerr << "Require three following parameters (input_dtype, quantized_dtype, granularity)"
            << std::endl;
  std::cerr << "                            ";
  std::cerr << "Ex: --quantize_dequantize_weights float32 uint8 channel" << std::endl;
  std::cerr << std::endl;
}

int entry(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "ERROR: Failed to parse arguments" << std::endl;
    std::cerr << std::endl;
    print_help(argv[0]);
    return 255;
  }

  // Simple argument parser (based on map)
  std::map<std::string, OptionHook> argparse;
  luci::CircleOptimizer optimizer;

  auto options = optimizer.options();

  // TODO use better parsing library (ex: boost.program_options)
  argparse["--quantize_dequantize_weights"] = [&options](const char **argv) {
    options->enable(Algorithms::QuantizeDequantizeWeights);

    if (argv[0] == nullptr || argv[1] == nullptr || argv[2] == nullptr)
      throw std::runtime_error(
          "--quantize_dequantize_weights must have three following parameters.");

    std::string input_dtype = argv[0];
    std::string output_dtype = argv[1];
    std::string granularity = argv[2];

    if (input_dtype.empty() || output_dtype.empty() || granularity.empty() ||
        input_dtype.substr(0, 2).compare("--") == 0 ||
        output_dtype.substr(0, 2).compare("--") == 0 || granularity.substr(0, 2).compare("--") == 0)
      throw std::runtime_error("Wrong algorithm parameters for --quantize_dequantize_weights.");

    options->param(AlgorithmParameters::Quantize_input_dtype, input_dtype);
    options->param(AlgorithmParameters::Quantize_output_dtype, output_dtype);
    options->param(AlgorithmParameters::Quantize_granularity, granularity);
    return 3;
  };

  // TODO use better parsing library (ex: boost.program_options)
  argparse["--quantize_with_minmax"] = [&options](const char **argv) {
    options->enable(Algorithms::QuantizeWithMinMax);

    if (argv[0] == nullptr || argv[1] == nullptr || argv[2] == nullptr)
      throw std::runtime_error("--quantize_with_minmax must have three following parameters.");

    std::string input_dtype = argv[0];
    std::string output_dtype = argv[1];
    std::string granularity = argv[2];

    if (input_dtype.empty() || output_dtype.empty() || granularity.empty() ||
        input_dtype.substr(0, 2).compare("--") == 0 ||
        output_dtype.substr(0, 2).compare("--") == 0 || granularity.substr(0, 2).compare("--") == 0)
      throw std::runtime_error("Wrong algorithm parameters for --quantize_with_minmax.");

    options->param(AlgorithmParameters::Quantize_input_dtype, input_dtype);
    options->param(AlgorithmParameters::Quantize_output_dtype, output_dtype);
    options->param(AlgorithmParameters::Quantize_granularity, granularity);
    return 3;
  };

  for (int n = 1; n < argc - 2; ++n)
  {
    const std::string tag{argv[n]};
    auto it = argparse.find(tag);
    if (it == argparse.end())
    {
      std::cerr << "Option '" << tag << "' is not supported" << std::endl;
      std::cerr << std::endl;
      print_help(argv[0]);
      return 255;
    }

    n += it->second((const char **)&argv[n + 1]);
  }

  std::string input_path = argv[argc - 2];
  std::string output_path = argv[argc - 1];

  // Load model from the file
  foder::FileLoader file_loader{input_path};
  std::vector<char> model_data = file_loader.load();
  const circle::Model *circle_model = circle::GetModel(model_data.data());
  if (circle_model == nullptr)
  {
    std::cerr << "ERROR: Failed to load circle '" << input_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  // Import from input Circle file
  luci::Importer importer;
  auto module = importer.importModule(circle_model);

  for (size_t idx = 0; idx < module->size(); ++idx)
  {
    auto graph = module->graph(idx);

    // quantize the graph
    optimizer.quantize(graph);

    if (!luci::validate(graph))
    {
      std::cerr << "ERROR: Quantized graph is invalid" << std::endl;
      return 255;
    }
  }

  // Export to output Circle file
  luci::CircleExporter exporter;

  CircleExpContract contract(module.get(), output_path);

  if (!exporter.invoke(&contract))
  {
    std::cerr << "ERROR: Failed to export '" << output_path << "'" << std::endl;
    return 255;
  }

  return 0;
}
