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

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <regex>

class circle_input_names_test : public ::testing::Test
{
protected:
  // Override Test::SetUp method to run before each test starts
  void SetUp(void) override;

protected:
  // Helper functions for tests
  std::vector<std::string> get_input_names_of(std::string op);

protected:
  // Dictionary string containing all input names of each op in JSON format
  std::string _input_names_dict_str;
};

void circle_input_names_test::SetUp(void)
{
  std::string cmd = std::getenv("CIRCLE_INPUT_NAMES_PATH");
  if (cmd.empty())
  {
    throw std::runtime_error("CIRCLE_INPUT_NAMES_PATH is not found");
  }

  FILE *fp = popen(cmd.c_str(), "r");
  if (!fp)
  {
    throw std::runtime_error("popen() failed");
  }

  char buff[1024];
  std::string result = "";
  try
  {
    while (fgets(buff, sizeof(buff), fp) != NULL)
    {
      result += buff;
    }
  }
  catch (...)
  {
    pclose(fp);
    throw;
  }

  _input_names_dict_str = result;
  pclose(fp);

  return;
}

std::vector<std::string> circle_input_names_test::get_input_names_of(std::string op)
{
  std::vector<std::string> input_names{};

  // Find op string key from _input_names_dict_str and parse the values input input_names vector
  size_t pos = _input_names_dict_str.find(op);
  if (pos == std::string::npos)
  {
    return input_names;
  }
  else
  {
    std::string substr = _input_names_dict_str.substr(pos);
    size_t start_pos = substr.find("[");
    size_t end_pos = substr.find("]");
    if (start_pos != std::string::npos && end_pos != std::string::npos)
    {
      std::string names = substr.substr(start_pos + 1, end_pos - start_pos - 1);
      std::stringstream ss(names);
      std::string name;
      while (std::getline(ss, name, ','))
      {
        std::smatch match;
        std::regex pattern = std::regex(R"(^\s*\"([^\"]+)\"\s*$)");
        if (std::regex_match(name, match, pattern))
        {
          input_names.push_back(match[1].str());
        }
      }
    }
  }
  return std::move(input_names);
}

TEST_F(circle_input_names_test, valid_command) { ASSERT_FALSE(_input_names_dict_str.empty()); }

TEST_F(circle_input_names_test, valid_names_softmax)
{
  // "SOFTMAX" should have single input: "logits"
  auto names = get_input_names_of("SOFTMAX");
  ASSERT_EQ(names.size(), 1);
  ASSERT_EQ(names[0], "logits");
}

TEST_F(circle_input_names_test, valid_names_conv2d)
{
  // "CONV_2D" should have three inputs: "input", "filter", "bias"
  auto names = get_input_names_of("CONV_2D");
  ASSERT_EQ(names.size(), 3);
  ASSERT_EQ(names[0], "input");
  ASSERT_EQ(names[1], "filter");
  ASSERT_EQ(names[2], "bias");
}

TEST_F(circle_input_names_test, not_exist_opname_NEG)
{
  // names of "NOT_EXIST_OP" should be empty
  auto names = get_input_names_of("NOT_EXIST_OP");
  ASSERT_EQ(names.size(), 0);
}

TEST_F(circle_input_names_test, lower_case_opname_NEG)
{
  // Upper case opname should be used
  auto names = get_input_names_of("conv_2d");
  ASSERT_EQ(names.size(), 0);
}

TEST_F(circle_input_names_test, out_of_bounds_NEG)
{
  auto names = get_input_names_of("CONV_2D");
  // names[3] should throw exception since it's out of bounds
  EXPECT_ANY_THROW(names.at(3));
}
