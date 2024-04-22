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

namespace
{
/**
 * @brief Tokenize given string
 *
 * Assumes given string looks like below.
 *
 * - '1,2,5,7,9'
 * - '1-5,6,7,9,12-14'
 * - 'tensor_a,tensor_b,tensor_d'
 *
 * NOTE. 1-5 is same with '1,2,3,4,5'.
 *
 * WARNING. SelectType::NAME doesn't allow '-' like 'tensor_a-tensor_c'.
 */
std::vector<std::string> split_into_vector(const std::string &str, const char &delim)
{
  std::vector<std::string> ret;
  std::istringstream is(str);
  for (std::string item; std::getline(is, item, delim);)
  {
    ret.push_back(item);
  }

  // Remove empty string
  ret.erase(std::remove_if(ret.begin(), ret.end(), [](const std::string &s) { return s.empty(); }),
            ret.end());

  return ret;
}

bool is_number(const std::string &s)
{
  return !s.empty() && std::find_if(s.begin(), s.end(),
                                    [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

bool is_number(const std::vector<std::string> &vec)
{
  for (const auto &s : vec)
  {
    if (not ::is_number(s))
    {
      return false;
    }
  }
  return true;
}

std::vector<uint32_t> parse_id(const std::string &str)
{
  std::vector<uint32_t> by_id;
  auto colon_tokens = ::split_into_vector(str, ',');

  for (const auto &comma_token : colon_tokens)
  {
    auto dash_tokens = ::split_into_vector(comma_token, '-');
    if (not ::is_number(dash_tokens))
    {
      throw std::runtime_error{
        "ERROR: To select operator by id, please use these args: [0-9], '-', ','"};
    }

    // Convert string into integer
    std::vector<uint32_t> int_tokens;
    try
    {
      std::transform(dash_tokens.begin(), dash_tokens.end(), std::back_inserter(int_tokens),
                     [](const std::string &str) { return static_cast<uint32_t>(std::stoi(str)); });
    }
    catch (const std::out_of_range &)
    {
      // Uf input is big integer like '123467891234', stoi throws this exception.
      throw std::runtime_error{"ERROR: Argument is out of range."};
    }
    catch (...)
    {
      throw std::runtime_error{"ERROR: Unknown error"};
    }

    switch (int_tokens.size())
    {
      case 0: // inputs like "-"
      {
        throw std::runtime_error{"ERROR: Nothing was entered"};
      }
      case 1: // inputs like "1", "2"
      {
        by_id.push_back(int_tokens.at(0));
        break;
      }
      case 2: // inputs like "1-2", "11-50"
      {
        for (uint32_t i = int_tokens.at(0); i <= int_tokens.at(1); i++)
        {
          by_id.push_back(i);
        }
        break;
      }
      default: // inputs like "1-2-3"
      {
        throw std::runtime_error{"ERROR: Too many '-' in str."};
      }
    }
  }
  return by_id;
}

} // namespace

int entry(int argc, char **argv)
{
  arser::Arser arser("circle training configure provides training graph model for current model");

  arser.add_argument("input").help("Input circle model");
  arser.add_argument("output").help("Output circle model");
  arser.add_argument("--by_id").help("Input operation id to select nodes.");

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

  if (!arser["--by_id"])
  {
    std::cerr << "ERROR: Either option '--by_id' " << std::endl;
    std::cerr << arser;
    return EXIT_FAILURE;
  }

  auto operator_input = arser.get<std::string>("--by_id");

  std::vector<uint32_t> div_ids = parse_id(operator_input);

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
  auto training_graph = gen_graph.createTrainingGraph(div_ids);

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

  return 0;
}
