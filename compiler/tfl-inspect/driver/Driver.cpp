/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "Dump.h"

#include <stdex/Memory.h>

#include <functional>
#include <iostream>
#include <map>
#include <vector>
#include <string>

using OptionHook = std::function<std::unique_ptr<tflinspect::DumpInterface>(void)>;

int entry(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "ERROR: Failed to parse arguments" << std::endl;
    std::cerr << std::endl;
    std::cerr << "USAGE: " << argv[0] << " [options] [tflite]" << std::endl;
    std::cerr << "   --operators : dump operators in tflite file" << std::endl;
    std::cerr << "   --conv2d_weight : dump Conv2D series weight operators in tflite file"
              << std::endl;
    std::cerr << "   --op_version : dump operator version of tflite" << std::endl;
    return 255;
  }

  // Simple argument parser (based on map)
  std::map<std::string, OptionHook> argparse;

  argparse["--operators"] = [&](void) {
    // dump all operators
    return std::move(stdex::make_unique<tflinspect::DumpOperators>());
  };

  argparse["--conv2d_weight"] = [&](void) {
    // dump Conv2D, DepthwiseConv2D weight operators
    return std::move(stdex::make_unique<tflinspect::DumpConv2DWeight>());
  };

  argparse["--op_version"] = [&](void) {
    // dump Conv2D, DepthwiseConv2D weight operators
    return std::move(stdex::make_unique<tflinspect::DumpOperatorVersion>());
  };

  std::vector<std::unique_ptr<tflinspect::DumpInterface>> dumps;

  for (int n = 1; n < argc - 1; ++n)
  {
    const std::string tag{argv[n]};

    auto it = argparse.find(tag);
    if (it == argparse.end())
    {
      std::cerr << "Option '" << tag << "' is not supported" << std::endl;
      return 255;
    }
    auto dump = it->second();
    assert(dump != nullptr);
    dumps.push_back(std::move(dump));
  }

  std::string model_file = argv[argc - 1];

  // Load TF lite model from a tflite file
  auto model = tflinspect::load_tflite(model_file);
  if (model == nullptr)
  {
    std::cerr << "ERROR: Failed to load tflite '" << model_file << "'" << std::endl;
    return 255;
  }

  const tflite::Model *tflmodel = model->model();
  if (tflmodel == nullptr)
  {
    std::cerr << "ERROR: Failed to load tflite '" << model_file << "'" << std::endl;
    return 255;
  }

  for (auto &dump : dumps)
  {
    dump->run(std::cout, tflmodel);
  }

  return 0;
}
