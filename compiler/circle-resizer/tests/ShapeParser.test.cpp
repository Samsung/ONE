/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ShapeParser.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

using namespace circle_resizer;

class ParseShapeTestFixture
  : public ::testing::TestWithParam<std::tuple<std::string, std::vector<Shape>>>
{
};

TEST_P(ParseShapeTestFixture, proper_shape_returned)
{
  const auto [input_shape_str, expected_shapes] = GetParam();
  std::vector<Shape> result_shapes;
  EXPECT_NO_THROW(result_shapes = parse_shapes(input_shape_str));
  ASSERT_EQ(result_shapes, expected_shapes);
}

INSTANTIATE_TEST_SUITE_P(
  ParseShapeTest, ParseShapeTestFixture,
  ::testing::Values(
    // single shape
    std::make_tuple("[3,4]", std::vector<Shape>{Shape{Dim{3}, Dim{4}}}),
    std::make_tuple("[3]", std::vector<Shape>{Shape{Dim{3}}}),
    std::make_tuple("[-1]", std::vector<Shape>{Shape{Dim{-1}}}),
    std::make_tuple("[  5,  6]", std::vector<Shape>{Shape{Dim{5}, Dim{6}}}),
    std::make_tuple("[3 , 4]", std::vector<Shape>{Shape{Dim{3}, Dim{4}}}),
    std::make_tuple("[-1 , 4]", std::vector<Shape>{Shape{Dim{-1}, Dim{4}}}),
    // many shapes
    std::make_tuple("[3,4],[5,6]",
                    std::vector<Shape>{Shape{Dim{3}, Dim{4}}, Shape{Dim{5}, Dim{6}}}),
    std::make_tuple("[1],[2]", std::vector<Shape>{Shape{Dim{1}}, Shape{Dim{2}}}),
    std::make_tuple(" [3, 4] , [5,6]",
                    std::vector<Shape>{Shape{Dim{3}, Dim{4}}, Shape{Dim{5}, Dim{6}}}),
    std::make_tuple(" [ 1   ] ,[  2]", std::vector<Shape>{Shape{Dim{1}}, Shape{Dim{2}}}),
    std::make_tuple(" [ 1   ] ,[  2]  ", std::vector<Shape>{Shape{Dim{1}}, Shape{Dim{2}}}),
    std::make_tuple(" [1,2],[3,4,5],[6,7,8,9]",
                    std::vector<Shape>{Shape{Dim{1}, Dim{2}}, Shape{Dim{3}, Dim{4}, Dim{5}},
                                       Shape{Dim{6}, Dim{7}, Dim{8}, Dim{9}}})));

class InvalidArgParseShapeTestFixture : public ::testing::TestWithParam<std::string>
{
};

TEST_P(InvalidArgParseShapeTestFixture, invalid_input)
{
  EXPECT_THROW(parse_shapes(GetParam()), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(InvalidArgParseShape, InvalidArgParseShapeTestFixture,
                         ::testing::Values(std::string{""}, std::string{"]"}, std::string{"1a,2"},
                                           std::string{"[-2]"}, std::string{"[-2,1,3]"},
                                           std::string{"-1"}, std::string{"7,7"}, std::string{"8"},
                                           std::string{"[8],9"}, std::string{"1,2"},
                                           std::string{"[1],[2],"}));
