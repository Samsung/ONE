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

#include "Dump.h"

#include <foder/FileLoader.h>

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using OptionHook = std::function<std::unique_ptr<circletensordump::DumpInterface>(void)>;

void print_help(const char *progname)
{
  std::cerr << "USAGE: " << progname << " options circle [hdf5]" << std::endl;
  std::cerr << "   --tensors : dump tensors in circle file" << std::endl;
  std::cerr << "   --tensors_to_hdf5 : dump tensors in circle file to hdf5 file" << std::endl;
}

int entry(int argc, char **argv)
{
  if (argc < 3)
  {
    std::cerr << "ERROR: Failed to parse arguments" << std::endl;
    std::cerr << std::endl;
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  // Simple argument parser (based on map)
  std::map<std::string, OptionHook> argparse;

  argparse["--tensors"] = [&](void) {
    // dump all tensors
    return std::move(std::make_unique<circletensordump::DumpTensors>());
  };

  argparse["--tensors_to_hdf5"] = [&](void) {
    // dump all tensors to hdf5 file
    return std::move(std::make_unique<circletensordump::DumpTensorsToHdf5>());
  };

  std::unique_ptr<circletensordump::DumpInterface> dump;

  const std::string tag{argv[1]};
  auto it = argparse.find(tag);

  std::string model_file = argv[2];
  dump = std::move(it->second());
  std::string output_path;

  if (tag == "--tensors")
  {
    // DO NOTHING
  }
  else if (tag == "--tensors_to_hdf5")
  {
    if (argc != 4)
    {
      std::cerr << "Option '" << tag << "' needs hdf5 output path" << std::endl;
      print_help(argv[0]);
      return EXIT_FAILURE;
    }
    output_path = argv[3];
  }
  else
  {
    std::cerr << "Option '" << tag << "' is not supported" << std::endl;
    print_help(argv[0]);
    return EXIT_FAILURE;
  }

  // Load Circle model from a circle file
  foder::FileLoader fileLoader{model_file};
  std::vector<char> modelData = fileLoader.load();
  const circle::Model *circleModel = circle::GetModel(modelData.data());
  if (circleModel == nullptr)
  {
    std::cerr << "ERROR: Failed to load circle '" << model_file << "'" << std::endl;
    return EXIT_FAILURE;
  }

  dump->run(std::cout, circleModel, output_path);

  return EXIT_SUCCESS;
}
