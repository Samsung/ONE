/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

// from https://github.com/Samsung/ONE/blob/
// 7be192529936563e0910ae903f43a9eb97fc9214/compiler/arser/tests/arser.test.cpp
// NOTE renamed "BasicTest" to "ArserTest" for test suite name
// NOTE copied mostly negative tests

#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "arser/arser.h"

#include "arser_prompt.h"

using namespace arser;

TEST(ArserTest, option)
{
  /* arrange */
  Arser arser;

  arser.add_argument("--verbose")
    .nargs(0)
    .help("It provides additional details as to what the executable is doing");

  test::Prompt prompt("./executable --verbose");
  /* act */
  arser.parse(prompt.argc(), prompt.argv());
  /* assert */
  EXPECT_TRUE(arser["--verbose"]);
  EXPECT_TRUE(arser.get<bool>("--verbose"));
}

TEST(ArserTest, OptionalArgument)
{
  /* arrange */
  Arser arser;

  arser.add_argument("--volume")
    .nargs(1)
    .type(arser::DataType::INT32)
    .help("Set a volume as you provided.");
  arser.add_argument("--frequency")
    .nargs(1)
    .type(arser::DataType::FLOAT)
    .help("Set a frequency as you provided.");

  test::Prompt prompt("./radio --volume 5 --frequency 128.5");
  /* act */
  arser.parse(prompt.argc(), prompt.argv());
  /* assert */
  EXPECT_TRUE(arser["--volume"]);
  EXPECT_EQ(5, arser.get<int>("--volume"));

  EXPECT_TRUE(arser["--frequency"]);
  EXPECT_FLOAT_EQ(128.5, arser.get<float>("--frequency"));

  EXPECT_FALSE(arser["--price"]);
  EXPECT_THROW(arser.get<bool>("--volume"), std::runtime_error);
}

TEST(ArserTest, NonRequiredOptionalArgument_NEG)
{
  /* arrange */
  Arser arser;

  arser.add_argument("--weight")
    .nargs(1)
    .type(arser::DataType::INT32)
    .help("Set a volume as you provided.");

  test::Prompt prompt("./radio"); // empty argument
  /* act */
  arser.parse(prompt.argc(), prompt.argv());
  /* assert */
  EXPECT_FALSE(arser["--volume"]);
  EXPECT_THROW(arser.get<int>("--weight"), std::runtime_error);
}

TEST(ArserTest, RequiredOptionalArgument_NEG)
{
  /* arrange */
  Arser arser;

  arser.add_argument("--volume")
    .nargs(1)
    .type(arser::DataType::INT32)
    .required()
    .help("Set a volume as you provided.");

  test::Prompt prompt("./radio");
  /* act */ /* assert */
  EXPECT_THROW(arser.parse(prompt.argc(), prompt.argv()), std::runtime_error);
}

void printVersion(std::string version) { std::cerr << "arser version : " << version << std::endl; }

TEST(ArserTest, ExitWithFunctionCallWithBind)
{
  /* arrange */
  Arser arser;

  arser.add_argument("--version")
    .help("Show version and exit")
    .exit_with(std::bind(printVersion, "1.2.0"));

  test::Prompt prompt("./arser --version");
  /* act */ /* assert */
  EXPECT_EXIT(arser.parse(prompt.argc(), prompt.argv()), testing::ExitedWithCode(0),
              "arser version : 1.2.0");
}

TEST(ArserTest, ExitWithFunctionCallWithLamda)
{
  /* arrange */
  Arser arser;

  arser.add_argument("--shutdown").help("Shut down your computer").exit_with([](void) {
    std::cerr << "Good bye.." << std::endl;
  });

  arser.add_argument("OS").nargs(1).type(arser::DataType::STR).help("The OS you want to boot");

  test::Prompt prompt("./computer --shutdown");
  /* act */ /* assert */
  EXPECT_EXIT(arser.parse(prompt.argc(), prompt.argv()), testing::ExitedWithCode(0), "Good bye..");
}

TEST(ArserTest, DefaultValue)
{
  /* arrange */
  Arser arser;

  arser.add_argument("--delivery")
    .nargs(3)
    .type(arser::DataType::STR_VEC)
    .default_value("pizza", "chicken", "hamburger")
    .help("Enter three foods that you want to deliver");
  arser.add_argument("--assistant")
    .type(arser::DataType::STR)
    .default_value("Bixby")
    .help("Enter name of your assistant");
  arser.add_argument("--sound")
    .type(arser::DataType::BOOL)
    .nargs(1)
    .default_value(true)
    .help("Sound on/off");
  arser.add_argument("--number")
    .type(arser::DataType::INT32_VEC)
    .nargs(4)
    .default_value(1, 2, 3, 4)
    .help("Enter the number that you want to call");
  arser.add_argument("--floats")
    .type(arser::DataType::FLOAT_VEC)
    .nargs(2)
    .default_value(2.0f, 3.0f)
    .help("Enter the float number that you want to call");
  arser.add_argument("--time")
    .type(arser::DataType::INT32_VEC)
    .nargs(3)
    .default_value(0, 0, 0)
    .help("Current time(H/M/S)");
  arser.add_argument("--name")
    .type(arser::DataType::STR)
    .nargs(1)
    .default_value("no name")
    .help("Enter your name");

  test::Prompt prompt("/phone --time 1 52 34 --name arser");
  /* act */
  arser.parse(prompt.argc(), prompt.argv());
  /* assert */
  // 3 strings, no argument
  std::vector<std::string> delivery = arser.get<std::vector<std::string>>("--delivery");
  EXPECT_EQ("pizza", delivery.at(0));
  EXPECT_EQ("chicken", delivery.at(1));
  EXPECT_EQ("hamburger", delivery.at(2));
  // 1 string, no argument
  EXPECT_EQ("Bixby", arser.get<std::string>("--assistant"));
  // 1 bool, no argument
  EXPECT_EQ(true, arser.get<bool>("--sound"));
  // 4 integer, no argument
  std::vector<int> number = arser.get<std::vector<int>>("--number");
  EXPECT_EQ(1, number.at(0));
  EXPECT_EQ(2, number.at(1));
  EXPECT_EQ(3, number.at(2));
  EXPECT_EQ(4, number.at(3));
  // 2 float, no argument
  std::vector<float> floats = arser.get<std::vector<float>>("--floats");
  EXPECT_EQ(2.0f, floats.at(0));
  EXPECT_EQ(3.0f, floats.at(1));
  // 3 integer, 3 arguments
  std::vector<int> time = arser.get<std::vector<int>>("--time");
  EXPECT_EQ(1, time.at(0));
  EXPECT_EQ(52, time.at(1));
  EXPECT_EQ(34, time.at(2));
  // 1 string, 1 argument
  EXPECT_EQ("arser", arser.get<std::string>("--name"));
}

TEST(ArserTest, OptWithRequiredDuplicate_NEG)
{
  /* arrange */
  Arser arser;

  arser.add_argument("--input_path", "-i", "--input", "--in")
    .nargs(1)
    .type(arser::DataType::STR)
    .help("input path of this program.")
    .required();
  arser.add_argument("--output_path", "-o")
    .nargs(1)
    .type(arser::DataType::STR)
    .help("output path of this program.")
    .required(true);

  test::Prompt prompt("./driver --in /I/am/in.put -o I/am/out.put -i /I/am/duplicate");
  /* act */ /* assert */
  EXPECT_THROW(arser.parse(prompt.argc(), prompt.argv()), std::runtime_error);
}

TEST(ArserTest, AccumulateScalarOptions_WrongType_NEG)
{
  /* arrange */
  Arser arser;

  arser.add_argument("--specify").nargs(1).accumulated(true).type(arser::DataType::FLOAT);

  test::Prompt prompt("./driver --specify 1 --specify 2");
  /* act */
  arser.parse(prompt.argc(), prompt.argv());
  /* assert */
  EXPECT_TRUE(arser["--specify"]);

  EXPECT_THROW(arser.get<float>("--specify"), std::runtime_error);
}

/* Additional tests not from ONE for more test coverage */

TEST(ArserTest, OptionalArgument_NEG)
{
  /* arrange */
  Arser arser;

  EXPECT_THROW(arser.add_argument("-"), std::runtime_error);
  EXPECT_THROW(arser.add_argument("--"), std::runtime_error);
}

TEST(ArserTest, PrintUsage)
{
  /* arrange */
  Arser arser("program_description");

  arser.add_argument("--input_path", "-i", "--input", "--in")
    .nargs(2)
    .type(arser::DataType::STR)
    .help("input path of this program.")
    .required();

  arser.add_argument("the_path").help("The file");

  /* act */
  EXPECT_NO_THROW(std::cout << arser);
}

void print_version(void) { std::cout << "1.2.3.4"; }

TEST(ArserTest, UseHelper)
{
  /* arrange */
  Arser arser;

  arser.add_argument("the_path").help("The file");
  EXPECT_NO_THROW(Helper::add_version(arser, print_version));
  EXPECT_NO_THROW(Helper::add_verbose(arser));

  test::Prompt prompt("./driver --help");
  /* act */
  EXPECT_NO_THROW(arser.parse(prompt.argc(), prompt.argv()));
}

TEST(ArserTest, PositionalRequired_NEG)
{
  /* arrange */
  Arser arser("program_description");

  arser.add_argument("the_path").help("The file");

  test::Prompt prompt("./driver");
  /* act */
  EXPECT_THROW(arser.parse(prompt.argc(), prompt.argv()), std::runtime_error);
}

TEST(ArserTest, AccumulateScalarOptions_WrongVectorType_NEG)
{
  /* arrange */
  Arser arser;

  arser.add_argument("--specify").nargs(1).accumulated(true).type(arser::DataType::FLOAT);

  test::Prompt prompt("./driver --specify 1 --specify 2");
  /* act */
  arser.parse(prompt.argc(), prompt.argv());
  /* assert */
  EXPECT_TRUE(arser["--specify"]);
  std::vector<float> dummy_float;
  EXPECT_NO_THROW(arser.get_impl<float>("--specify", &dummy_float));

  std::vector<std::string> dummy_str;
  EXPECT_THROW(arser.get_impl<std::string>("--invalid", &dummy_str), std::runtime_error);
  EXPECT_THROW(arser.get_impl<std::string>("--specify", &dummy_str), std::runtime_error);
}
