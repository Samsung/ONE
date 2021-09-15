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

#include <foder/FileLoader.h>

#include <luci/Importer.h>
#include <luci/CircleOptimizer.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/UserSettings.h>

#include <oops/InternalExn.h>
#include <arser/arser.h>
#include <vconone/vconone.h>

#include <functional>
#include <iostream>
#include <map>
#include <string>

using OptionHook = std::function<int(const char **)>;

using Algorithms = luci::CircleOptimizer::Options::Algorithm;
using AlgorithmParameters = luci::CircleOptimizer::Options::AlgorithmParameters;

void print_exclusive_options(void)
{
  std::cout << "Use only one of the 3 options below." << std::endl;
  std::cout << "    --quantize_dequantize_weights" << std::endl;
  std::cout << "    --quantize_with_minmax" << std::endl;
  std::cout << "    --requantize" << std::endl;
  std::cout << "    --force_quantparam" << std::endl;
}

void print_version(void)
{
  std::cout << "circle-quantizer version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int entry(int argc, char **argv)
{
  // Simple argument parser (based on map)
  std::map<std::string, OptionHook> argparse;
  luci::CircleOptimizer optimizer;

  auto options = optimizer.options();
  auto settings = luci::UserSettings::settings();

  const std::string qdqw = "--quantize_dequantize_weights";
  const std::string qwmm = "--quantize_with_minmax";
  const std::string rq = "--requantize";
  const std::string fq = "--force_quantparam";

  const std::string gpd = "--generate_profile_data";

  arser::Arser arser("circle-quantizer provides circle model quantization");

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

  arser.add_argument(qdqw)
    .nargs(3)
    .type(arser::DataType::STR_VEC)
    .required(false)
    .help("Quantize-dequantize weight values required action before quantization. "
          "Three arguments required: input_dtype(float32) "
          "output_dtype(uint8) granularity(layer, channel)");

  arser.add_argument(qwmm)
    .nargs(3)
    .type(arser::DataType::STR_VEC)
    .required(false)
    .help("Quantize with min/max values. "
          "Three arguments required: input_dtype(float32) "
          "output_dtype(uint8) granularity(layer, channel)");

  arser.add_argument(rq)
    .nargs(2)
    .type(arser::DataType::STR_VEC)
    .required(false)
    .help("Requantize a quantized model. "
          "Two arguments required: input_dtype(int8) "
          "output_dtype(uint8)");

  arser.add_argument(fq)
    .nargs(3)
    .type(arser::DataType::STR_VEC)
    .required(false)
    .accumulated(true)
    .help("Write quantization parameters to the specified tensor. "
          "Three arguments required: tensor_name(string), "
          "scale(float) zero_point(int)");

  arser.add_argument("input").nargs(1).type(arser::DataType::STR).help("Input circle model");
  arser.add_argument("output").nargs(1).type(arser::DataType::STR).help("Output circle model");

  arser.add_argument(gpd).nargs(0).required(false).default_value(false).help(
    "This will turn on profiling data generation.");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cout << arser;
    return 255;
  }

  {
    // only one of qdqw, qwmm, rq, fq option can be used
    int32_t opt_used = arser[qdqw] ? 1 : 0;
    opt_used += arser[qwmm] ? 1 : 0;
    opt_used += arser[rq] ? 1 : 0;
    opt_used += arser[fq] ? 1 : 0;
    if (opt_used != 1)
    {
      print_exclusive_options();
      return 255;
    }
  }

  if (arser.get<bool>("--verbose"))
  {
    // The third parameter of setenv means REPLACE.
    // If REPLACE is zero, it does not overwrite an existing value.
    setenv("LUCI_LOG", "100", 0);
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

  if (arser[rq])
  {
    auto values = arser.get<std::vector<std::string>>(rq);
    if (values.size() != 2)
    {
      std::cerr << arser;
      return 255;
    }
    options->enable(Algorithms::Requantize);

    options->param(AlgorithmParameters::Quantize_input_dtype, values.at(0));
    options->param(AlgorithmParameters::Quantize_output_dtype, values.at(1));
  }

  if (arser[fq])
  {
    auto values = arser.get<std::vector<std::vector<std::string>>>(fq);

    std::vector<std::string> tensors;
    std::vector<std::string> scales;
    std::vector<std::string> zero_points;

    for (auto const value : values)
    {
      if (value.size() != 3)
      {
        std::cerr << arser;
        return 255;
      }

      tensors.push_back(value[0]);
      scales.push_back(value[1]);
      zero_points.push_back(value[2]);
    }

    options->enable(Algorithms::ForceQuantParam);

    options->params(AlgorithmParameters::Quantize_tensor_names, tensors);
    options->params(AlgorithmParameters::Quantize_scales, scales);
    options->params(AlgorithmParameters::Quantize_zero_points, zero_points);
  }

  std::string input_path = arser.get<std::string>("input");
  std::string output_path = arser.get<std::string>("output");

  if (arser[gpd])
    settings->set(luci::UserSettings::Key::ProfilingDataGen, true);

  // Load model from the file
  foder::FileLoader file_loader{input_path};
  std::vector<char> model_data = file_loader.load();

  // Verify flatbuffers
  flatbuffers::Verifier verifier{reinterpret_cast<uint8_t *>(model_data.data()), model_data.size()};
  if (!circle::VerifyModelBuffer(verifier))
  {
    std::cerr << "ERROR: Invalid input file '" << input_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

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

  luci::CircleFileExpContract contract(module.get(), output_path);

  if (!exporter.invoke(&contract))
  {
    std::cerr << "ERROR: Failed to export '" << output_path << "'" << std::endl;
    return 255;
  }

  return 0;
}
