/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "TensorInfoParser.cpp"
#include "nnkit/support/tftestinfo/ParsedTensor.h"

#include <gtest/gtest.h>

#include <memory>

using namespace nnkit::support::tftestinfo;

namespace
{

struct TensorInfo
{
  std::string line;
  ParsedTensor::Kind kind;
  std::string name;
  TF_DataType dtype;
  uint32_t rank;
  uint32_t dim[2];
};

TEST(NNKIT_TF_PARSER, success_case)
{
  // clang-format off
  TensorInfo tc_list[] = {
    {"input, in/placeholder_1:0, TF_FLOAT, [3, 2] # correct case",
     ParsedTensor::Kind::Input, "in/placeholder_1:0", TF_FLOAT, 2, {3, 2} },

    {"output, aa/bb.cc:0, TF_FLOAT, []", // empty shape
     ParsedTensor::Kind::Output, "aa/bb.cc:0", TF_FLOAT, 0, {0, 0} },

    {"output, aa:0, TF_FLOAT, [] # this is a comment", // string with comment
     ParsedTensor::Kind::Output, "aa:0", TF_FLOAT, 0, {0, 0} },

    {"output, ...:0, TF_FLOAT, [] # this is a comment", // name test. TF works with this name
     ParsedTensor::Kind::Output, "...:0", TF_FLOAT, 0, {0, 0} },
  };
  // clang-format on

  for (auto tc : tc_list)
  {
    std::unique_ptr<ParsedTensor> tensor = parse_line(tc.line);

    ASSERT_EQ(tensor->kind(), tc.kind);
    ASSERT_EQ(tensor->name(), tc.name);
    ASSERT_EQ(tensor->dtype(), tc.dtype);
    ASSERT_EQ(tensor->shape().rank(), tc.rank);
    for (int d = 0; d < tc.rank; d++)
      ASSERT_EQ(tensor->shape().dim(d), tc.dim[d]);
  }
}

TEST(NNKIT_TF_PARSER, failure_case)
{
  // clang-format off
  std::string exception_list[] = {
     "WRONG_KIND, a:0, TF_FLOAT, [3, 2]",
     "input, a:0, WRONG_TYPE, [3, 2]",
     "input, a:0, TF_FLOAT, 3, 2", // missing bracelets
     "input, a:0, TF_FLOAT,, [3, 2]", // wrong commas, wrong bracelets
     "input, a:0, TF_FLOAT,, [3, 2,]", // wrong commas
     "a:0, TF_FLOAT, [3, 2]", // missing kind
     "input, TF_FLOAT, [3, 2]", // missing name
     "input, a:0, [3, 2]", // missing type
     "input, aa:0, TF_FLOAT", // missing shape
     "input, aa, TF_FLOAT, [abc]", // wrong name
     "input, a$a:0, TF_FLOAT, [abc]", // wrong name
     "input, aa:a, TF_FLOAT, [abc]", // wrong name (wrong value index)
     "input aa:a, TF_FLOAT, [1]", // missing comma, exception.what() is "A line must be either 'input' or 'output' but : input aa:a"
     "input, aa:a TF_FLOAT, [1]", // missing comma
     "input, aa:a, TF_FLOAT [1]", // missing comma,
  };
  // clang-format on

  for (auto tc : exception_list)
  {
    try
    {
      parse_line(tc);
      FAIL();
    }
    catch (const std::exception &e)
    {
      std::cout << e.what() << '\n';
    }
  }
}

TEST(NNKIT_TF_PARSER, comment_case)
{
  // clang-format off
  std::string tc_list[] = {
     "", // empty line
     "# this is a comment",
  };
  // clang-format on

  for (auto tc : tc_list)
  {
    ASSERT_EQ(parse_line(tc), nullptr);
  }
}

} // namespace
