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
#include <cstring>
#include <fstream>
#include <vector>

class circle_interpreter_test : public ::testing::Test
{
protected:
  bool initialize(void);
  bool run(const std::string &command);

protected:
  bool compare(const std::string &file1, const std::string &file2);

protected:
  std::string _artifacts_path;
  std::string _circle_interpreter_path;
  std::string _result;
};

bool circle_interpreter_test::initialize(void)
{
  char *path = std::getenv("ARTIFACTS_PATH");
  if (path == nullptr)
  {
    std::cerr << "ARTIFACTS_PATH not found" << std::endl;
    return false;
  }
  _artifacts_path = path;

  path = std::getenv("CIRCLE_INTERPRETER_PATH");
  if (path == nullptr)
  {
    std::cerr << "ARTIFACTS_BIN_PATH not found" << std::endl;
    return false;
  }
  _circle_interpreter_path = path;

  return true;
}

bool circle_interpreter_test::run(const std::string &command)
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

bool circle_interpreter_test::compare(const std::string &file1, const std::string &file2)
{
  std::ifstream f1(file1.c_str(), std::ifstream::in | std::ifstream::binary);
  std::ifstream f2(file2.c_str(), std::ifstream::in | std::ifstream::binary);

  if (!f1.is_open() || !f2.is_open())
  {
    return false;
  }

  char *buffer1 = new char[1024]();
  char *buffer2 = new char[1024]();

  do
  {
    f1.read(buffer1, 1024);
    f2.read(buffer2, 1024);

    if (std::memcmp(buffer1, buffer2, 1024) != 0)
    {
      delete[] buffer1;
      delete[] buffer2;
      return false;
    }
  } while (f1.good() || f2.good());

  delete[] buffer1;
  delete[] buffer2;
  return true;
}

TEST_F(circle_interpreter_test, show_help_msg)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string command = _circle_interpreter_path + " -h";
  if (!run(command))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("Usage: ./circle-interpreter");
  ASSERT_NE(std::string::npos, pos);
}

TEST_F(circle_interpreter_test, valid_command)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string model = _artifacts_path + "/Conv2D_000.circle";
  std::string input_prefix = _artifacts_path + "/Conv2D_000.circle.input";
  std::string output_prefix = "/tmp/Conv2D_000.circle.output";
  std::string generated_output = output_prefix + "0";
  std::remove(generated_output.c_str());
  std::string command =
    _circle_interpreter_path + " " + model + " " + input_prefix + " " + output_prefix;
  if (!run(command))
  {
    FAIL();
    return;
  }

  std::string expected_output = _artifacts_path + "/Conv2D_000.circle.output0";

  if (!compare(generated_output, expected_output))
  {
    FAIL();
    return;
  }
}

TEST_F(circle_interpreter_test, invalid_option_NEG)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string model = _artifacts_path + "/Conv2D_000.circle";
  std::string command = _circle_interpreter_path + " " + model;
  if (!run(command))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("Invalid argument");
  ASSERT_NE(std::string::npos, pos);
}

TEST_F(circle_interpreter_test, not_existing_model_NEG)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string not_existing_model = _artifacts_path + "/non_exist_file.foo";
  std::string input_prefix = _artifacts_path + "/Conv2D_000.circle.input";
  std::string output_prefix = "/tmp/Conv2D_000.circle.output";
  std::remove(output_prefix.c_str());
  std::string command =
    _circle_interpreter_path + " " + not_existing_model + " " + input_prefix + " " + output_prefix;
  if (!run(command))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("Failed to load");
  ASSERT_NE(std::string::npos, pos);
}

TEST_F(circle_interpreter_test, invalid_input_prefix_NEG)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  std::string model = _artifacts_path + "/Conv2D_000.circle";
  std::string input_prefix = _artifacts_path + "/non_exist_filee.foo";
  std::string output_prefix = "/tmp/Conv2D_000.circle.output";
  std::remove(output_prefix.c_str());
  std::string command =
    _circle_interpreter_path + " " + model + " " + input_prefix + " " + output_prefix;
  if (!run(command))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("Cannot open file");
  ASSERT_NE(std::string::npos, pos);
}

// Case of given input data has less number of values than model's input
TEST_F(circle_interpreter_test, input_data_mismatch_NEG)
{
  if (!initialize())
  {
    FAIL();
    return;
  }

  // Conv2D_000.circle input: 1x3x3x2
  std::string model = _artifacts_path + "/Conv2D_000.circle";
  // Concatenation_000.circle input0: 1x4x4x1 input1: 1x4x4x2
  std::string input_prefix = _artifacts_path + "/Concatenation_000.circle.input";
  std::string output_prefix = "/tmp/Conv2D_000.circle.output";
  std::remove(output_prefix.c_str());
  std::string command =
    _circle_interpreter_path + " " + model + " " + input_prefix + " " + output_prefix;
  if (!run(command))
  {
    FAIL();
    return;
  }

  const auto pos = _result.find("Failed to read data from file");
  ASSERT_NE(std::string::npos, pos);
}
