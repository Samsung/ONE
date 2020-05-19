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

#include "Model.h"
#include "CircleExpContract.h"

#include <luci/Importer.h>
#include <luci/CircleOptimizer.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>

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
  std::cerr << "   --fuse_instnorm : Enable FuseInstanceNormalization Pass" << std::endl;
  std::cerr << "   --resolve_customop_batchmatmul : Enable ResolveCustomOpBatchMatMulPass Pass"
            << std::endl;
  std::cerr << "   --quantize_with_minmax : Enable QuantizeWithMinMax Pass" << std::endl;
  std::cerr
      << "                            Require two following parameters (input_dtype, output_dtype)"
      << std::endl;
  std::cerr << "                            Ex: --quantize_with_minmax float32 uint8" << std::endl;
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

  // TODO merge this with help message
  argparse["--fuse_instnorm"] = [&options](const char **) {
    options->enable(Algorithms::FuseInstanceNorm);
    return 0;
  };
  argparse["--resolve_customop_batchmatmul"] = [&options](const char **) {
    options->enable(Algorithms::ResolveCustomOpBatchMatMul);
    return 0;
  };
  argparse["--quantize_with_minmax"] = [&options](const char **argv) {
    options->enable(Algorithms::QuantizeWithMinMax);
    std::string input_dtype = argv[0];
    assert(!input_dtype.empty());
    std::string output_dtype = argv[1];
    assert(!output_dtype.empty());
    options->param(AlgorithmParameters::QuantizeWithMinMax_input_dtype, input_dtype);
    options->param(AlgorithmParameters::QuantizeWithMinMax_output_dtype, output_dtype);
    return 2;
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
  std::unique_ptr<luci::Model> model = luci::load_model(input_path);
  if (model == nullptr)
  {
    std::cerr << "ERROR: Failed to load '" << input_path << "'" << std::endl;
    return 255;
  }

  const circle::Model *input_model = model->model();
  if (input_model == nullptr)
  {
    std::cerr << "ERROR: Failed to read '" << input_path << "'" << std::endl;
    return 255;
  }

  // Import from input Circle file
  luci::Importer importer;
  auto module = importer.importModule(input_model);

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
