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

#include <foder/FileLoader.h>

#include <luci/Importer.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include "ExecutionPlanner.h"

#include <arser/arser.h>

#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>

int entry(int argc, char **argv)
{
  arser::Arser arser("circle_execution_plan provides model with execution plan meta information");

  arser.add_argument("input").help("Input circle model");
  arser.add_argument("output").help("Output circle model");
  arser.add_argument("--platform").default_value("linux").help("Platform name: linux mcu cmsisnn");
  arser.add_argument("--use_dsp")
    .nargs(1)
    .type(arser::DataType::BOOL)
    .required(false)
    .default_value(false)
    .help("Plan with or without dsp (now can be used only with cmsisnn)");
  arser.add_argument("--save_allocations")
    .nargs(1)
    .required(false)
    .default_value("")
    .help("Path for output JSON file to save memory allocation info. "
          "Note: path end of file should have 'tracealloc.json' (example path: "
          "'../exec_plan_info.tracealloc.json')");

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

  const std::string input_path = arser.get<std::string>("input");
  const std::string output_path = arser.get<std::string>("output");
  const std::string platform_name = arser.get<std::string>("--platform");
  const bool use_dsp = arser.get<bool>("--use_dsp");
  const std::string json_path = arser.get<std::string>("--save_allocations");

  if (platform_name != "cmsisnn" && use_dsp)
  {
    std::cerr << "ERROR: Now use_dsp can be used only with cmsisnn" << std::endl;
    return EXIT_FAILURE;
  }

  circle_planner::SupportedPlatformType platform_type;
  if (platform_name == "linux")
  {
    platform_type = circle_planner::SupportedPlatformType::LINUX;
  }
  else if (platform_name == "mcu")
  {
    platform_type = circle_planner::SupportedPlatformType::MCU;
  }
  else if (platform_name == "cmsisnn")
  {
    platform_type = circle_planner::SupportedPlatformType::CMSISNN;
  }
  else
  {
    std::cerr << "ERROR: Invalid platform name '" << platform_name << "'" << std::endl;
    return EXIT_FAILURE;
  }

  bool is_save_allocations = false;

  if (!json_path.empty())
  {
    is_save_allocations = true;
  }

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

  // Do main job
  circle_planner::ExecutionPlanner execution_planner(module->graph(), {platform_type, use_dsp});
  execution_planner.make_execution_plan();

  if (is_save_allocations)
    execution_planner.create_json_allocation_file(json_path);

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
