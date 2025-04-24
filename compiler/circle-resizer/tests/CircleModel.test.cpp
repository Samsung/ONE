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

#include "CircleModel.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <fstream>

using namespace circle_resizer;
using ::testing::HasSubstr;

namespace
{

bool compare_shapes(const std::vector<Shape> &current, const std::vector<Shape> &expected)
{
  if (current.size() != expected.size())
  {
    return false;
  }
  for (size_t i = 0; i < current.size(); ++i)
  {
    if (!(current[i] == expected[i]))
    {
      return false;
    }
  }
  return true;
}

} // namespace

class CircleModelTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    char *path = std::getenv("ARTIFACTS_PATH");
    if (path == nullptr)
    {
      throw std::runtime_error("environmental variable ARTIFACTS_PATH required for circle-resizer "
                               "tests was not not provided");
    }
    _test_models_dir = path;
  }

protected:
  std::string _test_models_dir;
};

TEST_F(CircleModelTest, proper_input_output_shapes)
{
  CircleModel circle_model(_test_models_dir + "/Add_000.circle");
  EXPECT_TRUE(compare_shapes(circle_model.input_shapes(),
                             std::vector<Shape>{Shape{1, 4, 4, 3}, Shape{1, 4, 4, 3}}));
  EXPECT_TRUE(compare_shapes(circle_model.output_shapes(), std::vector<Shape>{Shape{1, 4, 4, 3}}));
}

TEST_F(CircleModelTest, proper_output_stream)
{
  CircleModel circle_model(_test_models_dir + "/Add_000.circle");
  std::stringstream out_stream;
  circle_model.save(out_stream);
  out_stream.seekg(0, std::ios::end);
  EXPECT_TRUE(out_stream.tellg() > 0);
}

TEST_F(CircleModelTest, model_file_not_exist_NEG)
{
  auto file_name = "/not_existed.circle";
  try
  {
    CircleModel circle_model(file_name);
    FAIL() << "Expected std::runtime_error";
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Failed to open file"));
    EXPECT_THAT(err.what(), HasSubstr(file_name));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}

TEST_F(CircleModelTest, invalid_model_NEG)
{
  try
  {
    CircleModel(std::vector<uint8_t>{1, 2, 3, 4, 5});
    FAIL() << "Expected std::runtime_error";
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Verification of the model failed"));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}

TEST_F(CircleModelTest, incorrect_output_stream_NEG)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/Add_000.circle");
  std::ofstream out_stream;
  try
  {
    circle_model->save(out_stream);
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Failed to write to output stream"));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}
