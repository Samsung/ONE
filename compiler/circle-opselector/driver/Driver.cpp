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

#include "OpSelector.h"
#include "Split.h"

#include <foder/FileLoader.h>

#include <luci/Importer.h>
#include <luci/CircleExporter.h>
#include <luci/CircleFileExpContract.h>
#include <luci/Import/CircleReader.h>

#include <luci/Profile/CircleNodeID.h>

#include <arser/arser.h>
#include <vconone/vconone.h>

#include <iostream>
#include <string>
#include <vector>

void print_version(void)
{
  std::cout << "circle-opselector version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

bool check_input(const std::string str)
{
  bool check_hyphen = false;

  if (not str.size())
    return false;
  if (str.at(0) == '-' || str[str.size() - 1] == '-')
  {
    std::cout << "Invalid input." << std::endl;
    return false;
  }

  for (char c : str)
  {
    if ('0' <= c && c <= '9')
      continue;
    else if (check_hyphen && c == '-') // when user enter '-' more than 2.
    {
      std::cout << "Too many '-' in str." << std::endl;
      return false;
    }
    else if (c == '-')
      check_hyphen = true;
    else // when user enter not allowed character, print alert msg.
    {
      std::cout << "To select operator by id, please use these args: [0-9], '-', ','" << std::endl;
      return false;
    }
  }
  return true;
}

void split_id_input(const std::string &str, std::vector<int> &by_id)
{
  std::istringstream ss;
  ss.str(str);
  std::string str_buf;

  while (getline(ss, str_buf, ','))
  {
    if (str_buf.length() && check_input(str_buf)) // input validation
    {
      try
      {
        if (str_buf.find('-') == std::string::npos) // if token has no '-'
          by_id.push_back(stoi(str_buf));
        else // tokenize again by '-'
        {
          std::istringstream ss2(str_buf);
          std::string token;
          int from_to[2], top = 0;

          while (getline(ss2, token, '-'))
            from_to[top++] = stoi(token);

          for (int number = from_to[0]; number <= from_to[1]; number++)
            by_id.push_back(number);
        }
      }
      catch (std::invalid_argument &error)
      {
        std::cerr << "ERROR: [circle-opselector] Invalid argument.(stoi)" << std::endl;
        exit(EXIT_FAILURE);
      }
      catch (std::out_of_range)
      {
        std::cout << "ERROR: [circle-opselector] Argument is out of range(stoi)\n";
        exit(EXIT_FAILURE);
      }
      catch (...)
      {
        std::cout << "ERROR: [circle-opselector] Unknown error(stoi)\n";
        exit(EXIT_FAILURE);
      }
    }
    else // Input validation failed
    {
      std::cerr << "ERROR: [circle-opselector] Input validation failed" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

void split_name_input(const std::string &str, std::vector<std::string> &by_name)
{
  std::istringstream ss;
  ss.str(str);
  std::string str_buf;

  while (getline(ss, str_buf, ','))
    by_name.push_back(str_buf);
}

int entry(int argc, char **argv)
{
  // TODO Add new option names!

  arser::Arser arser("circle-opselector provides selecting operations in circle model");

  arser.add_argument("--version")
    .nargs(0)
    .required(false)
    .default_value(false)
    .help("Show version information and exit")
    .exit_with(print_version);

  // TODO Add new options!

  arser.add_argument("input").nargs(1).type(arser::DataType::STR).help("Input circle model");
  arser.add_argument("output").nargs(1).type(arser::DataType::STR).help("Output circle model");

  // select option
  arser.add_argument("--by_id")
    .nargs(1)
    .type(arser::DataType::STR)
    .help("Input operation id to select nodes.");
  arser.add_argument("--by_name")
    .nargs(1)
    .type(arser::DataType::STR)
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

  std::string operator_input;

  std::vector<int> by_id;
  std::vector<std::string> by_name;

  std::string op;
  std::vector<int> oplist;
  bool select_mode = false;

  if (!arser["--by_id"] && !arser["--by_name"] || arser["--by_id"] && arser["--by_name"])
  {
    std::cout << "Either option '--by_id' or '--by_name' must be specified" << std::endl;
    std::cout << arser;
    return EXIT_FAILURE;
  }
  if (arser["--by_id"])
  {
    operator_input = arser.get<std::string>("--by_id");
    split_id_input(operator_input, by_id);
    //split_id(operator_input, by_id);
  }
  if (arser["--by_name"])
  {
    operator_input = arser.get<std::string>("--by_name");
    split_name_input(operator_input, by_name);
  }

  // option parsing test code.
  for (int x : by_id)
    std::cout << "by_id: " << x << std::endl;

  for (std::string line : by_name)
    std::cout << "by_name: " << line << std::endl;

  // Load model from the file
  foder::FileLoader file_loader{input_path};
  std::vector<char> model_data = file_loader.load();

  // Verify flatbuffers
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

  // Select and Import from user input.
  auto selector = std::make_unique<opselector::OpSelector>(circle_model);
  std::vector<const luci::CircleNode *> selected_nodes;

  // put selected nodes into map.
  if (by_id.size())
  {
    loco::Graph *graph = module.get()->graph(0); // get main subgraph.

    for (auto node : loco::all_nodes(graph))
    {
      auto cnode = loco::must_cast<const luci::CircleNode *>(node);

      try
      {
        auto node_id = luci::get_node_id(cnode); // if the node is not operator, throw runtime_error

        for (auto selected_id : by_id)
          if (selected_id == node_id) // find the selected id
            selected_nodes.emplace_back(cnode);
      }
      catch (std::runtime_error)
      {
        continue;
      }
    }
  }
  if (by_name.size())
  {
    loco::Graph *graph = module.get()->graph(0); // get main subgraph.

    for (auto node : loco::all_nodes(graph))
    {
      auto cnode = loco::must_cast<const luci::CircleNode *>(node);
      std::string node_name = cnode->name();

      try
      {
        luci::get_node_id(cnode); // if the node is not operator, throw runtime_error

        for (auto selected_name : by_name)
          if (selected_name.compare(node_name) == 0) // find the selected id
            selected_nodes.emplace_back(cnode);
      }
      catch (std::runtime_error)
      {
        continue;
      }
    }
  }
  // Import selected nodes.
  module = selector->select_nodes(selected_nodes);

  // Export to output Circle file
  luci::CircleExporter exporter;

  luci::CircleFileExpContract contract(module.get(), output_path);

  if (!exporter.invoke(&contract))
  {
    std::cerr << "ERROR: Failed to export '" << output_path << "'" << std::endl;
    return EXIT_FAILURE;
  }

  return 0;
}
