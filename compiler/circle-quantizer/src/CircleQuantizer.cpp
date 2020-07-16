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
#include <arser/arser.h>

#include <functional>
#include <iostream>
#include <map>
#include <string>

using OptionHook = std::function<int(const char **)>;

using Algorithms = luci::CircleOptimizer::Options::Algorithm;
using AlgorithmParameters = luci::CircleOptimizer::Options::AlgorithmParameters;

int entry(int argc, char **argv)
{
  // Simple argument parser (based on map)
  std::map<std::string, OptionHook> argparse;
  luci::CircleOptimizer optimizer;

  auto options = optimizer.options();

  const std::string qdqw = "--quantize_dequantize_weights";
  const std::string qwmm = "--quantize_with_minmax";

  arser::Arser arser("circle-quantizer provides circle model quantization");

  arser.add_argument(qdqw)
      .nargs(3)
      .type(arser::DataType::STR_VEC)
      .required(false)
      .help("Quantize-dequantize weight values required action before quantization. "
            "Three arguments required: input_dtype(float32) "
            "output_dtype(uint8) granularity(layer)");

  arser.add_argument(qwmm)
      .nargs(3)
      .type(arser::DataType::STR_VEC)
      .required(false)
      .help("Quantize with min/max values. "
            "Three arguments required: input_dtype(float32) "
            "output_dtype(uint8) granularity(layer)");

  arser.add_argument("input").nargs(1).type(arser::DataType::STR).help("Input circle model");
  arser.add_argument("output").nargs(1).type(arser::DataType::STR).help("Output circle model");

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

  if (arser[qdqw])
  {
    auto values = arser.get<std::vector<std::string>>(qdqw);
    if (values.size() != 3)
    {
      std::cerr << arser;
      return 255;
    }
    options->enable(Algorithms::QuantizeDequantizeWeights);

    options->param(AlgorithmParameters::Quantize_input_dtype, values.at(0));
    options->param(AlgorithmParameters::Quantize_output_dtype, values.at(1));
    options->param(AlgorithmParameters::Quantize_granularity, values.at(2));
  }

  if (arser[qwmm])
  {
    auto values = arser.get<std::vector<std::string>>(qwmm);
    if (values.size() != 3)
    {
      std::cerr << arser;
      return 255;
    }
    options->enable(Algorithms::QuantizeWithMinMax);

    options->param(AlgorithmParameters::Quantize_input_dtype, values.at(0));
    options->param(AlgorithmParameters::Quantize_output_dtype, values.at(1));
    options->param(AlgorithmParameters::Quantize_granularity, values.at(2));
  }

  std::string input_path = arser.get<std::string>("input");
  std::string output_path = arser.get<std::string>("output");

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
