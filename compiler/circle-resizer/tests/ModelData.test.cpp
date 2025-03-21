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

#include "ModelData.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>

using namespace circle_resizer;
using ::testing::HasSubstr;

class ModelDataTest : public ::testing::Test
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

TEST_F(ModelDataTest, neg_model_file_not_exist)
{
  auto file_name = "/not_existed.circle";
  try
  {
    ModelData model_data(file_name);
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

TEST_F(ModelDataTest, neg_invalid_model)
{
  try
  {
    ModelData(std::vector<uint8_t>{1, 2, 3, 4, 5});
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

TEST_F(ModelDataTest, neg_incorrect_output_stream)
{
  auto model_data = std::make_shared<ModelData>(_test_models_dir + "/Add_000.circle");
  std::ofstream out_stream;
  try
  {
    model_data->save(out_stream);
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
