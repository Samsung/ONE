/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "support/CommandLine.h"

#include "gtest/gtest.h"

using namespace nnc::cli;

void soption_checker1(const Option<std::string> &opt) { ASSERT_EQ(opt, "SOME_VALUE1,SOME_VALUE2"); }

void soption_checker2(const Option<std::string> &opt) { ASSERT_EQ(opt, "AAA_VALUE"); }

void boption_checker(const Option<bool> &opt) { ASSERT_EQ(opt, false); }

/**
 * declare command line options for testing
 */
//
// declare string option
//
// test option with several names
Option<std::string> SMultiOpt(optname("--multopt, -m, -mul"),
                              overview("description of option with several names"));
// test option with single name
Option<std::string> SSingleOpt(optname("--single"),
                               overview("description of option with single name"));
// test option with several separators
Option<std::string> SSeveralSepOpt(optname("-several_separators"),
                                   overview("description of option with several separators"), "",
                                   optional(false), optvalues(""), soption_checker1,
                                   separators("=, :"));
// test option with one separator
Option<std::string> SOneSepOpt(optname("--one_separarot"),
                               overview("description of option with one separator"), "",
                               optional(false), optvalues(""), soption_checker2, separators("="));
// test option with defalut value
Option<std::string> SDefaultOpt(optname("-default_val_opt"),
                                overview("description of option with default value"),
                                "DEFAULT_VALUE");
// test optional option
Option<std::string> SOptionalOpt(optname("--optional_opt"),
                                 overview("description of optional option"), "", optional(true));
// test option with valid values
Option<std::string> SValidValsOpt(optname("-valid_opt"),
                                  overview("description of option with valid values"), "",
                                  optional(true), optvalues("value1, value2"));

//
// declare integer options
//
// test option with negative value and valid values
Option<int32_t> NNegOpt(optname("-neg_val"),
                        overview("description of integer option with negative value"), -1,
                        optional(false), optvalues("-42, -33"));

// test option with default negative value
Option<int32_t>
  NDefaultNegOpt(optname("-default_neg_val"),
                 overview("description of integer option with default negative value"), -33);
// test option with positive values
Option<uint32_t> NPosOpt(optname("-pos_val"),
                         overview("description of integer option with positive value"), 1,
                         optional(false), optvalues("42, 33"));

//
// declare char options
//
Option<char> CharOpt(optname("-char-opt"), overview("description of char option"), '\0',
                     optional(false), optvalues("a, b"));

Option<char> DashOpt(optname("-dash_opt"), overview("description of char option with dash value"),
                     '\0', optional(false), optvalues("-"));

//
// declare bool option
//
Option<bool> BoolOpt(optname("-bool_opt"), overview("description of bool option"), true,
                     optional(false), optvalues(""), boption_checker, separators("="));
Option<bool> BoolOpt2(optname("-bool-opt2"), overview("description of bool option with value"));

//
// declare vector<string> option
//
Option<std::vector<std::string>> VecStrOpt1(optname("-vec_opt1"),
                                            overview("description of vector option"));
Option<std::vector<std::string>> VecStrOpt2(optname("-vec_opt2"),
                                            overview("description of vector option"));
Option<std::vector<std::string>> VecStrOptWithVals(optname("--vec_opt_with_vals"),
                                                   overview("description of vector option"),
                                                   std::vector<std::string>(), optional(false),
                                                   optvalues("abc, 123, xxx"));
//
// declare options in group
//
//
// declare bool option
//
Option<bool> GroupOpt1(optname("-group_opt1"), overview("description of group option"), true,
                       optional(false), optvalues(""), nullptr, separators(""), showopt(true),
                       IOption::Group::caffe2);
Option<std::string> GroupOpt2(optname("-group_opt2"), overview("description of group option"),
                              std::string(), optional(true), optvalues(""), nullptr, separators(""),
                              showopt(true), IOption::Group::caffe2);
Option<int32_t> GroupOpt3(optname("-group_opt3"), overview("description of group option"), 42,
                          optional(true), optvalues(""), nullptr, separators(""), showopt(true),
                          IOption::Group::onnx);

// test options
TEST(SUPPORT_NNC, verify_cl_options)
{
  // create command line
  const char *argv[] = {
    "CLTest", // program name
    // string options
    "-m", "multiopt_value",                        // second name for option with several names
    "--single", "single_value",                    // option with single name
    "-several_separators:SOME_VALUE1,SOME_VALUE2", // test option with several separators
    "--one_separarot=AAA_VALUE",                   // test option whit one separator
    "-default_val_opt",                            // test option with default value
    "--optional_opt", "/home/guest/tmp",           // test optional option
    "-valid_opt", "value2",                        // test options with defined values
    // integer options
    "-neg_val", "-42",  // test negative value for integer option
    "-default_neg_val", // test integer option with default value
    "-pos_val", "33",   // test positive value for integer option
    // char options
    "-char-opt", "b", "-dash_opt", "-",
    // bool options
    "-bool_opt=false", "-bool-opt2",
    // vector of strings options
    "-vec_opt1", "1", "c", "222", "ABC", "857", "-vec_opt2", "--vec_opt_with_vals", "abc", "123",
    "xxx", "abc", "xxx",
    // grouped options
    "-group_opt1", "-group_opt2", "abc", "-group_opt3", "11", nullptr};
  int argc = (sizeof(argv) / sizeof(argv[0])) - 1;

  // It must be failed if option is not passed and other options are in the same group
  argv[argc - 5] = "-m"; // disable -group_opt1
  ASSERT_DEATH(CommandLine::getParser()->parseCommandLine(argc, argv), "");
  argv[argc - 5] = "-group_opt1"; // enable -group_opt1

  // test when mandatory grouped option is not passed. It must be OK if options being from the same
  // group are missed
  argv[argc - 4] = "-m"; // disable -group_opt2
  argv[argc - 2] = "-m"; // disable -group_opt3
  CommandLine::getParser()->parseCommandLine(argc, argv);
  argv[argc - 4] = "-group_opt2"; // enable -group_opt2
  argv[argc - 2] = "-group_opt3"; // enable -group_opt3

  // parse command line
  CommandLine::getParser()->parseCommandLine(argc, argv);

  // here we put value from options
  std::string tmp_string = SMultiOpt;
  int32_t tmp_sint = NNegOpt;
  char tmp_char = CharOpt;
  bool tmp_bool = BoolOpt;
  std::vector<std::string> tmp_vec = VecStrOpt1;

  //
  // string options
  //
  // check option with several names
  ASSERT_EQ(SMultiOpt, "multiopt_value");
  ASSERT_EQ(tmp_string, "multiopt_value");
  ASSERT_EQ(tmp_string, SMultiOpt);

  // check option with single name
  tmp_string = SSingleOpt;
  ASSERT_EQ(SSingleOpt, "single_value");
  ASSERT_EQ(tmp_string, "single_value");
  ASSERT_EQ(SSingleOpt, tmp_string);

  // check option with separators
  ASSERT_EQ(SSeveralSepOpt, "SOME_VALUE1,SOME_VALUE2");

  // check option with one separator
  ASSERT_EQ(SOneSepOpt, "AAA_VALUE");

  // check option with default value
  ASSERT_EQ(SDefaultOpt, "DEFAULT_VALUE");

  // check optional option
  ASSERT_EQ(SOptionalOpt, "/home/guest/tmp");

  //
  // integer options
  //
  // check option with valid values
  ASSERT_EQ(SValidValsOpt, "value2");

  // check option with negative value
  ASSERT_EQ(NNegOpt, -42);
  ASSERT_EQ(tmp_sint, -42);
  ASSERT_EQ(tmp_sint, NNegOpt);

  // check integer option with default value
  tmp_sint = NDefaultNegOpt;
  ASSERT_EQ(NDefaultNegOpt, -33);
  ASSERT_EQ(tmp_sint, -33);
  ASSERT_EQ(NDefaultNegOpt, tmp_sint);

  // check integer option with positive value
  ASSERT_EQ(NPosOpt, 33u);

  //
  // char options
  //
  ASSERT_EQ(CharOpt, 'b');
  ASSERT_EQ(tmp_char, 'b');
  ASSERT_EQ(tmp_char, CharOpt);

  tmp_char = DashOpt;
  ASSERT_EQ(DashOpt, '-');
  ASSERT_EQ(tmp_char, '-');
  ASSERT_EQ(DashOpt, tmp_char);

  //
  // bool options
  //
  ASSERT_EQ(BoolOpt, false);
  ASSERT_EQ(tmp_bool, false);
  ASSERT_EQ(tmp_bool, BoolOpt);

  tmp_bool = BoolOpt2;
  ASSERT_EQ(BoolOpt2, true);
  ASSERT_EQ(tmp_bool, true);
  ASSERT_EQ(BoolOpt2, tmp_bool);

  //
  // vector of strings options
  //
  ASSERT_EQ(tmp_vec, VecStrOpt1);
  ASSERT_EQ(VecStrOpt1[0], "1");
  ASSERT_EQ(tmp_vec[1], "c");
  ASSERT_EQ(VecStrOpt1[2], "222");
  ASSERT_EQ(tmp_vec[3], "ABC");
  ASSERT_EQ(VecStrOpt1[4], "857");

  tmp_vec = VecStrOpt2;
  ASSERT_EQ(VecStrOpt2, tmp_vec);
  ASSERT_TRUE(VecStrOpt2.empty());

  ASSERT_EQ(VecStrOptWithVals[0], "abc");
  ASSERT_EQ(VecStrOptWithVals[1], "123");
  ASSERT_EQ(VecStrOptWithVals[2], "xxx");
  ASSERT_EQ(VecStrOptWithVals[3], "abc");
  ASSERT_EQ(VecStrOptWithVals[4], "xxx");

  //
  // grouped options
  //
  ASSERT_TRUE(GroupOpt1.isGrouped() && GroupOpt2.isGrouped() && GroupOpt3.isGrouped());
  ASSERT_EQ(GroupOpt1.getGroup(), GroupOpt2.getGroup());
  ASSERT_NE(GroupOpt2.getGroup(), GroupOpt3.getGroup());
  ASSERT_EQ(GroupOpt3.getGroupName(), "onnx");
}
