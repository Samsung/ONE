/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ModuleIO.h"
#include "OpSelector.h"

#include <luci/ConnectNode.h>
#include <luci/Profile/CircleNodeID.h>
#include <luci/Service/CircleNodeClone.h>

#include <arser/arser.h>
#include <vconone/vconone.h>

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <numeric>
#include <sstream>

void print_version(void)
{
  std::cout << "circle-opselector version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

int entry(int argc, char **argv)
{
  // TODO Add new option names!

  arser::Arser arser("circle-opselector provides selecting operations in circle model");

  arser::Helper::add_version(arser, print_version);

  // TODO Add new options!

  arser.add_argument("input").help("Input circle model");
  arser.add_argument("output").help("Output circle model");

  // select option
  arser.add_argument("--by_id").nargs(1).accumulated(true).help(
    "Input operation id to select nodes.");
  arser.add_argument("--by_name")
    .nargs(1)
    .accumulated(true)
    .help("Input operation name to select nodes.");

  try
  {
    arser.parse(argc, argv);
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << err.what() << std::endl;
    std::cout << arser;
    return EXIT_FAILURE;
  }

  std::string input_path = arser.get<std::string>("input");
  std::string output_path = arser.get<std::string>("output");

  if (!arser["--by_id"] && !arser["--by_name"] || arser["--by_id"] && arser["--by_name"])
  {
    std::cerr << "ERROR: Either option '--by_id' or '--by_name' must be specified" << std::endl;
    std::cerr << arser;
    return EXIT_FAILURE;
  }

  // Import original circle file.
  auto module = opselector::getModule(input_path);

  // TODO support two or more subgraphs
  // if (module.get()->size() != 1)
  // {
  //   std::cerr << "ERROR: Not support two or more subgraphs" << std::endl;
  //   return EXIT_FAILURE;
  // }

  std::string select_type;
  if (arser["--by_id"])
  {
    select_type = "--by_id";
  }
  if (arser["--by_name"])
  {
    select_type = "--by_name";
  }

  const auto inputs = arser.get<std::vector<std::string>>(select_type);
  if (inputs.size() != module.get()->size())
  {
    std::cerr << "ERROR: The number of selected graphs should be same with the number of subgraphs "
                 "in the model"
              << std::endl;
    return EXIT_FAILURE;
  }

  opselector::OpSelector op_selector{module.get()};

  std::unique_ptr<luci::Module> new_module;

  if (arser["--by_id"])
  {
    new_module = op_selector.select_by<opselector::SelectType::ID>(inputs);
  }
  if (arser["--by_name"])
  {
    new_module = op_selector.select_by<opselector::SelectType::NAME>(inputs);
  }

  if (not opselector::exportModule(new_module.get(), output_path))
  {
    std::cerr << "ERROR: Cannot export the module" << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}
