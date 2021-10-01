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

void print_version(void)
{
  std::cout << "circle-opselector version " << vconone::get_string() << std::endl;
  std::cout << vconone::get_copyright() << std::endl;
}

/**
 * @brief  Validation function for user input
 *
 * @note   This function checks for inappropriate data in str.
 *         param str is the tokenized value of the user's input as ','.
 *         If str has values other than [0-9] and '-' or doesn't have
 *         data like '', it return false.
 *         And if str has multiple '-' such as '1--2', '1-2-3',
 *         or invalid '-' position like '-1', '1-', it return false too.
 */
bool check_input(const std::string &str)
{
  if (str.empty()) // if str is empty, exit.
    return false;
  if (str.at(0) == '-' || str[str.size() - 1] == '-') // if '-' is inappropriate
  {
    std::cerr << "ERROR: Invalid input. Please make sure - is between the numbers" << std::endl;
    return false;
  }

  bool has_hyphen = false; // '-' check flag. if '-' more than 2 in string, return false.

  for (char c : str)
  {
    if (isdigit(c)) // Make sure the data is in 0-9.
      continue;
    else if (has_hyphen && c == '-') // when user enter '-' more than 2.
    {
      std::cerr << "ERROR: Too many '-' in str." << std::endl;
      return false;
    }
    else if (c == '-') // found first '-' char.
      has_hyphen = true;
    else // when user enter not allowed character, print alert msg.
    {
      std::cerr << "ERROR: To select operator by id, please use these args: [0-9], '-', ','"
                << std::endl;
      return false;
    }
  }
  return true;
}

/**
 * @brief  Segmentation function for user's '--by_id' input
 *
 * @note   This function tokenizes the input data.
 *         First, divide it into ',', and if token has '-', devide it once more into '-'.
 *         For example, if user input is '12,34,56', it is devided into [12,34,56].
 *         If input is '1-2,34,56', it is devided into [[1,2],34,56].
 *         And '-' means range so, if input is '2-7', it means all integer between 2-7.
 */
void split_id_input(const std::string &str, std::vector<int> &by_id)
{
  std::istringstream ss;
  ss.str(str);
  std::string str_buf;

  while (getline(ss, str_buf, ','))
  {
    if (check_input(str_buf)) // input validation
    {
      try
      {
        if (str_buf.find('-') == std::string::npos) // if token has no '-'
          by_id.push_back(stoi(str_buf));
        else // tokenize again by '-'
        {
          std::istringstream ss2(str_buf);
          std::string token;
          int from_to[2], top = 0; // In input '1-5', put 1 in from_to[0], put 5 in from_to[1].

          // Because check_input handle multiple '-' inputs, top is less than 3.
          while (getline(ss2, token, '-'))
            from_to[top++] = stoi(token);

          for (int number = from_to[0]; number <= from_to[1]; number++)
            by_id.push_back(number);
        }
      }
      catch (std::out_of_range)
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
    }
    else // Input validation failed
    {
      std::cerr << "ERROR: Input validation failed. Please make sure your input is number."
                << std::endl;
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

  if (!arser["--by_id"] && !arser["--by_name"] || arser["--by_id"] && arser["--by_name"])
  {
    std::cerr << "ERROR: Either option '--by_id' or '--by_name' must be specified" << std::endl;
    std::cerr << arser;
    return EXIT_FAILURE;
  }

  if (arser["--by_id"])
  {
    operator_input = arser.get<std::string>("--by_id");
    split_id_input(operator_input, by_id);
  }
  if (arser["--by_name"])
  {
    operator_input = arser.get<std::string>("--by_name");
    split_name_input(operator_input, by_name);
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
