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
  std::cerr << "Optimization options: " << std::endl;
  std::cerr << "   --all : Enable all optimize options" << std::endl;
  std::cerr << "   --fuse_bcq : Enable FuseBCQ Pass" << std::endl;
  std::cerr << "   --fuse_instnorm : Enable FuseInstanceNormalization Pass" << std::endl;
  std::cerr << "   --resolve_customop_add : Enable ResolveCustomOpAddPass Pass" << std::endl;
  std::cerr << "   --resolve_customop_batchmatmul : Enable ResolveCustomOpBatchMatMulPass Pass"
            << std::endl;
  std::cerr << "   --resolve_customop_matmul : Enable ResolveCustomOpMatMulPass Pass" << std::endl;
  std::cerr << "Execution options:" << std::endl;
  std::cerr << "   --mute_warnings : Turn off warning messages" << std::endl;
  std::cerr << "   --disable_validation : Turn off operator vaidations" << std::endl;
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
  auto settings = luci::UserSettings::settings();

  // TODO merge this with help message
  argparse["--all"] = [&options](const char **) {
    options->enable(Algorithms::FuseBCQ);
    options->enable(Algorithms::FuseInstanceNorm);
    options->enable(Algorithms::ResolveCustomOpAdd);
    options->enable(Algorithms::ResolveCustomOpBatchMatMul);
    return 0;
  };
  argparse["--fuse_bcq"] = [&options](const char **) {
    options->enable(Algorithms::FuseBCQ);
    return 0;
  };
  argparse["--fuse_instnorm"] = [&options](const char **) {
    options->enable(Algorithms::FuseInstanceNorm);
    return 0;
  };
  argparse["--resolve_customop_add"] = [&options](const char **) {
    options->enable(Algorithms::ResolveCustomOpAdd);
    return 0;
  };
  argparse["--resolve_customop_batchmatmul"] = [&options](const char **) {
    options->enable(Algorithms::ResolveCustomOpBatchMatMul);
    return 0;
  };
  argparse["--resolve_customop_matmul"] = [&options](const char **) {
    options->enable(Algorithms::ResolveCustomOpMatMul);
    return 0;
  };

  argparse["--mute_warnings"] = [&settings](const char **) {
    settings->set(luci::UserSettings::Key::MuteWarnings, true);
    return 0;
  };
  argparse["--disable_validation"] = [&settings](const char **) {
    settings->set(luci::UserSettings::Key::DisableValidation, true);
    return 0;
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
