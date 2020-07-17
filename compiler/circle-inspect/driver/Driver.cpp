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

#include "Dump.h"

#include <foder/FileLoader.h>
#include <stdex/Memory.h>

#include <functional>
#include <iostream>
#include <map>
#include <vector>
#include <string>

using OptionHook = std::function<std::unique_ptr<circleinspect::DumpInterface>(void)>;

int entry(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "ERROR: Failed to parse arguments" << std::endl;
    std::cerr << std::endl;
    std::cerr << "USAGE: " << argv[0] << " [options] [circle]" << std::endl;
    std::cerr << "   --operators : dump operators in circle file" << std::endl;
    std::cerr << "   --conv2d_weight : dump Conv2D series weight operators in circle file"
              << std::endl;
    std::cerr << "   --op_version : dump operator version of circle" << std::endl;
    return 255;
  }

  // Simple argument parser (based on map)
  std::map<std::string, OptionHook> argparse;

  argparse["--operators"] = [&](void) {
    // dump all operators
    return std::move(stdex::make_unique<circleinspect::DumpOperators>());
  };

  argparse["--conv2d_weight"] = [&](void) {
    // dump Conv2D, DepthwiseConv2D weight operators
    return std::move(stdex::make_unique<circleinspect::DumpConv2DWeight>());
  };

  argparse["--op_version"] = [&](void) {
    // dump Conv2D, DepthwiseConv2D weight operators
    return std::move(stdex::make_unique<circleinspect::DumpOperatorVersion>());
  };

  std::vector<std::unique_ptr<circleinspect::DumpInterface>> dumps;

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

  // Load Circle model from a circle file
  foder::FileLoader fileLoader{model_file};
  std::vector<char> modelData = fileLoader.load();
  const circle::Model *circleModel = circle::GetModel(modelData.data());
  if (circleModel == nullptr)
  {
    std::cerr << "ERROR: Failed to load circle '" << model_file << "'" << std::endl;
    return 255;
  }

  for (auto &dump : dumps)
  {
    dump->run(std::cout, circleModel);
  }

  return 0;
}
