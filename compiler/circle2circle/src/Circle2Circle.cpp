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
#include <luci/UserSettings.h>

#include <stdex/Memory.h>
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
  auto settings = luci::UserSettings::settings();

  arser::Arser arser("circle2circle provides circle model optimization and transformations");

  arser.add_argument("--all").nargs(0).required(false).help("Enable all optimize options");

  arser.add_argument("--fuse_bcq")
      .nargs(0)
      .required(false)
      .help("This will fuse operators and apply Binary Coded Quantization");

  arser.add_argument("--fuse_instnorm")
      .nargs(0)
      .required(false)
      .help("This will fuse operators to InstanceNorm operator");

  arser.add_argument("--resolve_customop_add")
      .nargs(0)
      .required(false)
      .help("This will convert Custom(Add) to Add operator");

  arser.add_argument("--resolve_customop_batchmatmul")
      .nargs(0)
      .required(false)
      .help("This will convert Custom(BatchMatmul) to BatchMatmul operator");

  arser.add_argument("--resolve_customop_matmul")
      .nargs(0)
      .required(false)
      .help("This will convert Custom(Matmul) to Matmul operator");

  arser.add_argument("--mute_warnings")
      .nargs(0)
      .required(false)
      .help("This will turn off warning messages");

  arser.add_argument("--disable_validation")
      .nargs(0)
      .required(false)
      .help("This will turn off operator vaidations. May help input model investigation.");

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

  if (arser["--all"])
  {
    if (arser.get<bool>("--all"))
    {
      options->enable(Algorithms::FuseBCQ);
      options->enable(Algorithms::FuseInstanceNorm);
      options->enable(Algorithms::ResolveCustomOpAdd);
      options->enable(Algorithms::ResolveCustomOpBatchMatMul);
      options->enable(Algorithms::ResolveCustomOpMatMul);
    }
  }

  if (arser["--fuse_bcq"])
  {
    if (arser.get<bool>("--fuse_bcq"))
      options->enable(Algorithms::FuseBCQ);
  }
  if (arser["--fuse_instnorm"])
  {
    if (arser.get<bool>("--fuse_instnorm"))
      options->enable(Algorithms::FuseInstanceNorm);
  }
  if (arser["--resolve_customop_add"])
  {
    if (arser.get<bool>("--resolve_customop_add"))
      options->enable(Algorithms::ResolveCustomOpAdd);
  }
  if (arser["--resolve_customop_batchmatmul"])
  {
    if (arser.get<bool>("--resolve_customop_batchmatmul"))
      options->enable(Algorithms::ResolveCustomOpBatchMatMul);
  }
  if (arser["--resolve_customop_matmul"])
  {
    if (arser.get<bool>("--resolve_customop_matmul"))
      options->enable(Algorithms::ResolveCustomOpMatMul);
  }

  if (arser["--mute_warnings"])
  {
    if (arser.get<bool>("--mute_warnings"))
      settings->set(luci::UserSettings::Key::MuteWarnings, true);
  }
  if (arser["--disable_validation"])
  {
    if (arser.get<bool>("--disable_validation"))
      settings->set(luci::UserSettings::Key::DisableValidation, true);
  }

  std::string input_path = arser.get<std::string>("input");
  std::string output_path = arser.get<std::string>("output");

  // Load model from the file
  foder::FileLoader file_loader{input_path};
  std::vector<char> model_data;

  try
  {
    model_data = file_loader.load();
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
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

    // call luci optimizations
    optimizer.optimize(graph);

    if (!luci::validate(graph))
    {
      std::cerr << "ERROR: Optimized graph is invalid" << std::endl;
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
