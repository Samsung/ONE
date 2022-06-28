/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include <fstream>
#include <vector>

class cirlce_operator_test : public ::testing::Test
{
protected:
  bool initialize(void);
  bool run(const std::string &command);

protected:
  bool load(const std::string &file);

protected:
  std::string _artifacts_path;
  std::string _circle_operator_path;
  std::string _result;
};

bool cirlce_operator_test::initialize(void)
{
  char *path = std::getenv("ARTIFACTS_PATH");
  if (path == nullptr)
  {
    std::cerr << "ARTIFACTS_PATH not found" << std::endl;
    return false;
  }
  _artifacts_path = path;

  path = std::getenv("CIRCLE_OPERATOR_PATH");
  if (path == nullptr)
  {
    std::cerr << "ARTIFACTS_BIN_PATH not found" << std::endl;
    return false;
  }
  _circle_operator_path = path;

  return true;
}

bool cirlce_operator_test::run(const std::string &command)
{
  std::vector<char> buffer(260);
  std::string result = "";
  std::string cmd_err = command + " 2>&1";
  FILE *pipe = popen(cmd_err.c_str(), "r");
  if (!pipe)
  {
    return false;
  }
  try
  {
    while (fgets(&buffer[0], buffer.size(), pipe) != NULL)
    {
      result += &buffer[0];
    }
  }
  catch (...)
  {
    pclose(pipe);
    return false;
  }
  pclose(pipe);
  _result = result;

  std::cout << _result << std::endl;

  return true;
}

bool cirlce_operator_test::load(const std::string &file)
{
  std::ifstream tmp(file.c_str());
  if (tmp.fail())
    return false;

  std::stringstream buffer;
  buffer << tmp.rdbuf();
  _result = buffer.str();
  return true;
}

TEST_F(cirlce_operator_test, valid_names)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string model = _artifacts_path + "/Add_000.circle";
  std::string command = _circle_operator_path + " --name " + model;
  if (!run(command))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("ofm");
  ASSERT_NE(std::string::npos, pos);
}

TEST_F(cirlce_operator_test, valid_codes)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string model = _artifacts_path + "/Add_000.circle";
  std::string command = _circle_operator_path + " --code " + model;
  if (!run(command))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("ADD");
  ASSERT_NE(std::string::npos, pos);
}

TEST_F(cirlce_operator_test, invalid_option_NEG)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string model = _artifacts_path + "/Add_000.circle";
  std::string command = _circle_operator_path + " --opname " + model;
  if (!run(command))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("Invalid argument");
  ASSERT_NE(std::string::npos, pos);
}

TEST_F(cirlce_operator_test, check_code_name)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string model = _artifacts_path + "/Add_000.circle";
  std::string command = _circle_operator_path + " --code --name " + model;
  if (!run(command))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("ofm");
  ASSERT_NE(std::string::npos, pos);
  const auto pos2 = _result.find("ADD");
  ASSERT_NE(std::string::npos, pos2);
}

TEST_F(cirlce_operator_test, nonexist_file_NEG)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string model = _artifacts_path + "/non_exist_file.foo";
  std::string command = _circle_operator_path + " --name " + model;
  if (!run(command))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("ERROR");
  ASSERT_NE(std::string::npos, pos);
}

TEST_F(cirlce_operator_test, invalid_file_NEG)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string model = _artifacts_path + "/Add_000.recipe";
  std::string command = _circle_operator_path + " --name " + model;
  if (!run(command))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("ERROR");
  ASSERT_NE(std::string::npos, pos);
}

TEST_F(cirlce_operator_test, output_file)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string fileName("/tmp/a.txt");
  std::remove(fileName.c_str());
  std::string model = _artifacts_path + "/Add_000.circle";
  std::string command = _circle_operator_path + " --code --output_path " + fileName + " " + model;
  if (!run(command))
  {
    FAIL();
    return;
  }
  if (!load(fileName))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("ADD");
  ASSERT_NE(std::string::npos, pos);
}
