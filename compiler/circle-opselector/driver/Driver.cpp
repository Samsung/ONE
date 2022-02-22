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

#include <luci/Profile/CircleNodeID.h>

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

std::vector<std::string> split_into_vector(const std::string &str, const char &delim)
{
  std::vector<std::string> ret;
  std::istringstream is(str);
  for (std::string item; std::getline(is, item, delim);)
  {
    ret.push_back(item);
  }

  // remove empty string
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
    if (not::is_number(s))
    {
      return false;
    }
  }
  return true;
}

/**
 * @brief  Segmentation function for user's '--by_id' input
 *
 * @note   This function tokenizes the input data.s
 *         First, divide it into ',', and if token has '-', devide it once more into '-'.
 *         For example, if user input is '12,34,56', it is devided into [12,34,56].
 *         If input is '1-2,34,56', it is devided into [[1,2],34,56].
 *         And '-' means range so, if input is '2-7', it means all integer between 2-7.
 */
std::vector<uint32_t> split_id_input(const std::string &str)
{
  std::vector<uint32_t> by_id;

  // tokenize colon-separated string
  auto colon_tokens = ::split_into_vector(str, ',');
  if (colon_tokens.empty()) // input empty line like "".
  {
    std::cerr << "ERROR: Nothing was entered." << std::endl;
    exit(EXIT_FAILURE);
  }
  for (const auto &ctok : colon_tokens)
  {
    auto dash_tokens = ::split_into_vector(ctok, '-');
    if (not::is_number(dash_tokens))
    {
      std::cerr << "ERROR: To select operator by id, please use these args: [0-9], '-', ','"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    // convert string into integer
    std::vector<uint32_t> int_tokens;
    try
    {
      std::transform(dash_tokens.begin(), dash_tokens.end(), std::back_inserter(int_tokens),
                     [](const std::string &str) { return static_cast<uint32_t>(std::stoi(str)); });
    }
    catch (const std::out_of_range &)
    {
      // if input is big integer like '123467891234', stoi throw this exception.
      std::cerr << "ERROR: Argument is out of range." << std::endl;
      exit(EXIT_FAILURE);
    }
    catch (...)
    {
      std::cerr << "ERROR: Unknown error" << std::endl;
      exit(EXIT_FAILURE);
    }

    switch (int_tokens.size())
    {
      case 0: // inputs like "-"
      {
        std::cerr << "ERROR: Nothing was entered" << std::endl;
        exit(EXIT_FAILURE);
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
        std::cerr << "ERROR: Too many '-' in str." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
  }

  return by_id;
}

std::vector<std::string> split_name_input(const std::string &str)
{
  return ::split_into_vector(str, ',');
}

int entry(int argc, char **argv)
{
  // TODO Add new option names!

  arser::Arser arser("circle-opselector provides selecting operations in circle model");

  arser.add_argument("--version")
    .nargs(0)
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

  std::vector<uint32_t> by_id;
  std::vector<std::string> by_name;

  if (!arser["--by_id"] && !arser["--by_name"] || arser["--by_id"] && arser["--by_name"])
  {
    std::cerr << "ERROR: Either option '--by_id' or '--by_name' must be specified" << std::endl;
    std::cerr << arser;
    return EXIT_FAILURE;
  }

  if (arser["--by_id"])
  {
    operator_input = arser.get<std::string>("--by_id");
    by_id = split_id_input(operator_input);
  }
  if (arser["--by_name"])
  {
    operator_input = arser.get<std::string>("--by_name");
    by_name = split_name_input(operator_input);
  }

  // Import original circle file.
  auto module = opselector::getModule(input_path);

  // Select nodes from user input.
  std::vector<const luci::CircleNode *> selected_nodes;

  // put selected nodes into vector.
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

      for (auto selected_name : by_name)
        if (selected_name.compare(node_name) == 0) // find the selected name
          selected_nodes.emplace_back(cnode);
    }
  }
  if (selected_nodes.size() == 0)
  {
    std::cerr << "ERROR: No operator selected" << std::endl;
    exit(EXIT_FAILURE);
  }
  // TODO implement node selections

  // Export to output Circle file
  assert(opselector::exportModule(module.get(), output_path));

  return 0;
}
