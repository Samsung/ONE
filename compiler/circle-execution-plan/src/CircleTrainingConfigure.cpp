/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "GenerateTrainingGraph.h"

int entry(int argc, char **argv)
{
  arser::Arser arser("circle training configure provides training graph model for current model");

  arser.add_argument("input").help("Input circle model");
  arser.add_argument("output").help("Output circle model");

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

  // Generate training graph
  training_graph::GenerateTrainingGraph gen_graph(module.get());
  auto training_graph = gen_graph.createTrainingGraph();

  std::unique_ptr<luci::Module> module1 = std::make_unique<luci::Module>();
  module1->add(std::move(training_graph));

  // First export
  {
    // Export to output Circle file
    luci::CircleExporter exporter;
    luci::CircleFileExpContract contract(module1.get(), output_path);

    if (!exporter.invoke(&contract))
    {
      std::cerr << "ERROR: Failed to export '" << output_path << "'" << std::endl;
      return 255;
    }
  }

  // Import back to obtain indexes
  {
    foder::FileLoader file_loader_2{output_path};
    std::vector<char> model_data_2;

    try
    {
      model_data_2 = file_loader_2.load();
    }
    catch (const std::runtime_error &err)
    {
      std::cerr << err.what() << std::endl;
      return EXIT_FAILURE;
    }

    flatbuffers::Verifier verifier_2{reinterpret_cast<uint8_t *>(model_data_2.data()), model_data_2.size()};
    if (!circle::VerifyModelBuffer(verifier))
    {
      std::cerr << "ERROR: Invalid input file '" << input_path << "'" << std::endl;
      return EXIT_FAILURE;
    }

    const circle::Model *circle_model_2 = circle::GetModel(model_data_2.data());
    if (circle_model_2 == nullptr)
    {
      std::cerr << "ERROR: Failed to load circle '" << input_path << "'" << std::endl;
      return EXIT_FAILURE;
    }

    // Import from input Circle file
    auto module_2 = importer.importModule(circle_model_2);

    auto map_t_i = gen_graph.createMapTensorsIndexes(circle_model, circle_model_2);
    module_2->map_tenros_indexes(map_t_i);

    // Final export
    {
      // Export to output Circle file
      luci::CircleExporter exporter;
      luci::CircleFileExpContract contract(module_2.get(), output_path);

      if (!exporter.invoke(&contract))
      {
        std::cerr << "ERROR: Failed to export '" << output_path << "'" << std::endl;
        return 255;
      }
    }
  }
//
//  // Check metada
//  {
//    foder::FileLoader file_loader_2{output_path};
//    std::vector<char> model_data_2;
//
//    try
//    {
//      model_data_2 = file_loader_2.load();
//    }
//    catch (const std::runtime_error &err)
//    {
//      std::cerr << err.what() << std::endl;
//      return EXIT_FAILURE;
//    }
//
//    flatbuffers::Verifier verifier_2{reinterpret_cast<uint8_t *>(model_data_2.data()), model_data_2.size()};
//    if (!circle::VerifyModelBuffer(verifier))
//    {
//      std::cerr << "ERROR: Invalid input file '" << input_path << "'" << std::endl;
//      return EXIT_FAILURE;
//    }
//
//    const circle::Model *circle_model_2 = circle::GetModel(model_data_2.data());
//    if (circle_model_2 == nullptr)
//    {
//      std::cerr << "ERROR: Failed to load circle '" << input_path << "'" << std::endl;
//      return EXIT_FAILURE;
//    }
//
//    // Import from input Circle file
//    auto module_2 = importer.importModule(circle_model_2);
//    auto tmp = module_2->map_tenros_indexes();
//    std::cout << "tmp\n";
//
//  }


  return 0;
}
