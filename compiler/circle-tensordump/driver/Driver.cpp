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

#include <arser/arser.h>
#include <foder/FileLoader.h>

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

int entry(int argc, char **argv)
{
  arser::Arser arser{
      "circle-tensordump allows users to retrieve tensor information from a Circle model file"};

  arser.add_argument("circle").nargs(1).type(arser::DataType::STR).help("Circle file path to dump");
  arser.add_argument("--tensors").nargs(0).help("Dump to console");
  arser.add_argument("--tensors_to_hdf5")
      .nargs(1)
      .type(arser::DataType::STR)
      .help("Dump to hdf5 file. Specify hdf5 file path to be dumped");

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

  std::unique_ptr<circletensordump::DumpInterface> dump;

  std::string model_file = arser.get<std::string>("circle");
  std::string output_path;
  if (arser["--tensors_to_hdf5"])
  {
    dump = std::move(std::make_unique<circletensordump::DumpTensorsToHdf5>());
    output_path = arser.get<std::string>("--tensors_to_hdf5");
  }
  if (arser["--tensors"])
  {
    dump = std::move(std::make_unique<circletensordump::DumpTensors>());
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
