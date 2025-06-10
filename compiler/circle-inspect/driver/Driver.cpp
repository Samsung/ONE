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

#include <arser/arser.h>
#include <foder/FileLoader.h>

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <string>

int entry(int argc, char **argv)
{
  arser::Arser arser{
    "circle-inspect allows users to retrieve various information from a Circle model file"};
  arser.add_argument("--operators").nargs(0).help("Dump operators in circle file");
  arser.add_argument("--conv2d_weight")
    .nargs(0)
    .help("Dump Conv2D series weight operators in circle file");
  arser.add_argument("--constants").nargs(0).help("Dump constant tensors name");
  arser.add_argument("--op_version").nargs(0).help("Dump versions of the operators in circle file");
  arser.add_argument("--tensor_dtype").nargs(0).help("Dump dtype of tensors");
  arser.add_argument("--tensor_shape").nargs(0).help("Dump shape of tensors");
  arser.add_argument("circle").help("Circle file to inspect");

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

  if (!arser["--operators"] && !arser["--conv2d_weight"] && !arser["--op_version"] &&
      !arser["--tensor_dtype"] && !arser["--constants"] && !arser["--tensor_shape"])
  {
    std::cout << "At least one option must be specified" << std::endl;
    std::cout << arser;
    return 255;
  }

  std::vector<std::unique_ptr<circleinspect::DumpInterface>> dumps;

  if (arser["--operators"])
    dumps.push_back(std::make_unique<circleinspect::DumpOperators>());
  if (arser["--conv2d_weight"])
    dumps.push_back(std::make_unique<circleinspect::DumpConv2DWeight>());
  if (arser["--op_version"])
    dumps.push_back(std::make_unique<circleinspect::DumpOperatorVersion>());
  if (arser["--tensor_dtype"])
    dumps.push_back(std::make_unique<circleinspect::DumpTensorDType>());
  if (arser["--constants"])
    dumps.push_back(std::make_unique<circleinspect::DumpConstants>());
  if (arser["--tensor_shape"])
    dumps.push_back(std::make_unique<circleinspect::DumpTensorShape>());

  std::string model_file = arser.get<std::string>("circle");

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
    dump->run(std::cout, circleModel, &modelData);
  }

  return 0;
}
