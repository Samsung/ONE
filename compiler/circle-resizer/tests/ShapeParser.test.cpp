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
using Shapes = std::vector<Shape>;

class ParseShapeTestFixture : public ::testing::TestWithParam<std::tuple<std::string, Shapes>>
{
};

TEST_P(ParseShapeTestFixture, successful_parsing)
{
  const auto [input_shape_str, expected_shapes] = GetParam();
  Shapes result_shapes;
  EXPECT_NO_THROW(result_shapes = parse_shapes(input_shape_str));
  ASSERT_EQ(result_shapes, expected_shapes);
}

INSTANTIATE_TEST_SUITE_P(ParseShapeTest, ParseShapeTestFixture,
                         ::testing::Values(
                           // single shape
                           std::make_tuple("[3,4]", Shapes{Shape{3, 4}}),
                           std::make_tuple("[3]", Shapes{Shape{3}}),
                           std::make_tuple("[-1]", Shapes{Shape{Dim::dynamic()}}),
                           std::make_tuple("[  5,  6]", Shapes{Shape{5, 6}}),
                           std::make_tuple("[3 , 4]", Shapes{Shape{3, 4}}),
                           std::make_tuple("[-1 , 4]", Shapes{Shape{Dim::dynamic(), Dim{4}}}),
                           // many shapes
                           std::make_tuple("[3,4],[5,6]", Shapes{Shape{3, 4}, Shape{5, 6}}),
                           std::make_tuple("[1],[2]", Shapes{Shape{1}, Shape{2}}),
                           std::make_tuple(" [3, 4] , [5,6]", Shapes{Shape{3, 4}, Shape{5, 6}}),
                           std::make_tuple(" [ 1   ] ,[  2]", Shapes{Shape{1}, Shape{2}}),
                           std::make_tuple(" [ 1   ] ,[  2]  ", Shapes{Shape{1}, Shape{2}}),
                           std::make_tuple(" [1,2],[3,4,5],[6,7,8,9]",
                                           Shapes{Shape{1, 2}, Shape{3, 4, 5}, Shape{6, 7, 8, 9}}),
                           // scalars
                           std::make_tuple("[]", Shapes{Shape::scalar()}),
                           std::make_tuple("[],[]", Shapes{Shape::scalar(), Shape::scalar()}),
                           std::make_tuple("[],[2]", Shapes{Shape::scalar(), Shape{2}}),
                           std::make_tuple("[  ]", Shapes{Shape::scalar()})));

class InvalidArgParseShapeTestFixture : public ::testing::TestWithParam<std::string>
{
};

TEST_P(InvalidArgParseShapeTestFixture, invalid_input_NEG)
{
  EXPECT_THROW(parse_shapes(GetParam()), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(InvalidArgParseShape, InvalidArgParseShapeTestFixture,
                         ::testing::Values(std::string{""}, std::string{"]"}, std::string{"1a,2"},
                                           std::string{"[-2]"}, std::string{"[-2,1,3]"},
                                           std::string{"-1"}, std::string{"7,7"}, std::string{"8"},
                                           std::string{"[8],9"}, std::string{"1,2"},
                                           std::string{"[1],[2],"}, std::string{"[[]],"},
                                           std::string{"][1]"}, std::string{"]["}));
