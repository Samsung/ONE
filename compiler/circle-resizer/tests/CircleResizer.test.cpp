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

#include "CircleResizer.h"
#include "oops/UserExn.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace circle_resizer;
using ::testing::HasSubstr;

class CircleResizerTest : public ::testing::Test
{
protected:
  virtual void SetUp()
  {
    _test_models_dir = std::getenv("ARTIFACTS_PATH");
    assert(!_test_models_dir.empty());
  }

protected:
  std::string _test_models_dir;
};

TEST_F(CircleResizerTest, single_input_single_output)
{
  CircleResizer resizer(_test_models_dir + "/ExpandDims_000.circle");
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{4}, Dim{6}}};
  resizer.resize_model(new_input_shapes);
  EXPECT_EQ(resizer.input_shapes(), new_input_shapes);
  EXPECT_EQ(resizer.output_shapes(), (std::vector<Shape>{Shape{Dim{4}, Dim{1}, Dim{6}}}));
}

TEST_F(CircleResizerTest, single_input_two_outputs)
{
  CircleResizer resizer(_test_models_dir + "/CSE_Quantize_000.circle");
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{1}, Dim{6}, Dim{6}, Dim{4}}};
  resizer.resize_model(new_input_shapes);
  EXPECT_EQ(resizer.input_shapes(), new_input_shapes);
  EXPECT_EQ(resizer.output_shapes(), (std::vector<Shape>{Shape{Dim{1}, Dim{6}, Dim{6}, Dim{4}},
                                                         Shape{Dim{1}, Dim{6}, Dim{6}, Dim{4}}}));
}

TEST_F(CircleResizerTest, two_inputs_single_output)
{
  CircleResizer resizer(_test_models_dir + "/Add_000.circle");
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}},
                                                   Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}}};
  resizer.resize_model(new_input_shapes);
  EXPECT_EQ(resizer.input_shapes(), new_input_shapes);
  EXPECT_EQ(resizer.output_shapes(), (std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}}}));
}

TEST_F(CircleResizerTest, two_inputs_two_outputs)
{
  CircleResizer resizer(_test_models_dir + "/Part_Add_Sqrt_Rsqrt_000.circle");
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}},
                                                   Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}}};
  resizer.resize_model(new_input_shapes);
  EXPECT_EQ(resizer.input_shapes(), new_input_shapes);
  EXPECT_EQ(resizer.output_shapes(), (std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}},
                                                         Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}}}));
}

TEST_F(CircleResizerTest, neg_model_file_not_exist)
{
  auto file_name = "/not_existed.circle";
  try
  {
    CircleResizer(_test_models_dir + file_name);
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

TEST_F(CircleResizerTest, neg_invalid_model)
{
  try
  {
    CircleResizer(std::vector<uint8_t>{1, 2, 3, 4, 5});
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

TEST_F(CircleResizerTest, neg_not_all_input_shapes_provided)
{
  CircleResizer resizer(_test_models_dir + "/Add_000.circle");
  try
  {
    resizer.resize_model(std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}}});
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Expected input shapes: 2 while provided: 1"));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}

TEST_F(CircleResizerTest, neg_incorrect_rank_of_new_shape)
{
  CircleResizer resizer(_test_models_dir + "/ExpandDims_000.circle");
  try
  {
    resizer.resize_model(std::vector<Shape>{Shape{Dim{3}}});
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Provided shape rank: 1 is different from expected: 2"));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}

TEST_F(CircleResizerTest, neg_shape_inference_failed)
{
  CircleResizer resizer(_test_models_dir + "/DepthwiseConv2D_000.circle");
  EXPECT_THROW(resizer.resize_model(std::vector<Shape>{Shape{Dim{1}, Dim{64}, Dim{64}, Dim{8}},
                                                       Shape{Dim{1}, Dim{2}, Dim{2}, Dim{3}}}),
               oops::UserExn);
}

TEST_F(CircleResizerTest, save_model_without_change)
{
  CircleResizer resizer(_test_models_dir + "/ExpandDims_000.circle");
  std::stringstream out_stream;
  resizer.save_model(out_stream);
  const std::string &model_buf_str = out_stream.str();
  std::vector<uint8_t> model_buffer(std::begin(model_buf_str), std::end(model_buf_str));
  model_buffer.insert(std::end(model_buffer), std::begin(model_buf_str), std::end(model_buf_str));
  CircleResizer resizer2(model_buffer);
  EXPECT_EQ(resizer2.input_shapes(), (std::vector<Shape>{Shape{Dim{3}, Dim{3}}}));
  EXPECT_EQ(resizer2.output_shapes(), (std::vector<Shape>{Shape{Dim{3}, Dim{1}, Dim{3}}}));
}

TEST_F(CircleResizerTest, save_model_after_resizing)
{
  CircleResizer resizer(_test_models_dir + "/ExpandDims_000.circle");
  std::stringstream out_stream;
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{4}, Dim{6}}};
  resizer.resize_model(new_input_shapes);
  resizer.save_model(out_stream);
  const std::string &model_buf_str = out_stream.str();
  std::vector<uint8_t> model_buffer(std::begin(model_buf_str), std::end(model_buf_str));
  model_buffer.insert(std::end(model_buffer), std::begin(model_buf_str), std::end(model_buf_str));
  CircleResizer resizer2(model_buffer);
  EXPECT_EQ(resizer2.input_shapes(), new_input_shapes);
  EXPECT_EQ(resizer2.output_shapes(), (std::vector<Shape>{Shape{Dim{4}, Dim{1}, Dim{6}}}));
}

TEST_F(CircleResizerTest, neg_incorrect_output_stream)
{
  CircleResizer resizer(_test_models_dir + "/Add_000.circle");
  std::ofstream out_stream;
  try
  {
    resizer.save_model(out_stream);
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
