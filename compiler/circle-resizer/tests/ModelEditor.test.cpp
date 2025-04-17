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

#include "ModelEditor.h"
#include "oops/UserExn.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>

using namespace circle_resizer;
using ::testing::HasSubstr;

class ModelEditorTest : public ::testing::Test
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

TEST_F(ModelEditorTest, single_input_single_output)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{4}, Dim{6}}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(), (std::vector<Shape>{Shape{Dim{4}, Dim{1}, Dim{6}}}));
}

TEST_F(ModelEditorTest, single_input_two_outputs)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/CSE_Quantize_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{1}, Dim{6}, Dim{6}, Dim{4}}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(),
            (std::vector<Shape>{Shape{Dim{1}, Dim{6}, Dim{6}, Dim{4}},
                                Shape{Dim{1}, Dim{6}, Dim{6}, Dim{4}}}));
}

TEST_F(ModelEditorTest, two_inputs_single_output)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/Add_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}},
                                                   Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(),
            (std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}}}));
}

TEST_F(ModelEditorTest, two_inputs_two_outputs)
{
  auto circle_model =
    std::make_shared<CircleModel>(_test_models_dir + "/Part_Add_Sqrt_Rsqrt_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}},
                                                   Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(),
            (std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}},
                                Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}}}));
}

TEST_F(ModelEditorTest, neg_not_all_input_shapes_provided)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/Add_000.circle");
  ModelEditor editor(circle_model);
  try
  {
    editor.resize_inputs(std::vector<Shape>{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}}});
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

TEST_F(ModelEditorTest, neg_incorrect_rank_of_new_shape)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  try
  {
    editor.resize_inputs(std::vector<Shape>{Shape{Dim{3}}});
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

TEST_F(ModelEditorTest, save_without_change)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  std::stringstream out_stream;
  circle_model->save(out_stream);
  const std::string &model_buf_str = out_stream.str();
  std::vector<uint8_t> model_buffer(std::begin(model_buf_str), std::end(model_buf_str));
  model_buffer.insert(std::end(model_buffer), std::begin(model_buf_str), std::end(model_buf_str));
  auto circle_model_2 = std::make_shared<CircleModel>(model_buffer);
  ModelEditor editor_2(circle_model_2);
  EXPECT_EQ(circle_model_2->input_shapes(), (std::vector<Shape>{Shape{Dim{3}, Dim{3}}}));
  EXPECT_EQ(circle_model_2->output_shapes(), (std::vector<Shape>{Shape{Dim{3}, Dim{1}, Dim{3}}}));
}

TEST_F(ModelEditorTest, save_after_resizing)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  std::stringstream out_stream;
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{4}, Dim{6}}};
  editor.resize_inputs(new_input_shapes);
  circle_model->save(out_stream);
  const std::string &model_buf_str = out_stream.str();
  std::vector<uint8_t> model_buffer(std::begin(model_buf_str), std::end(model_buf_str));
  model_buffer.insert(std::end(model_buffer), std::begin(model_buf_str), std::end(model_buf_str));
  auto circle_model_2 = std::make_shared<CircleModel>(model_buffer);
  ModelEditor editor_2(circle_model_2);
  EXPECT_EQ(circle_model_2->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model_2->output_shapes(), (std::vector<Shape>{Shape{Dim{4}, Dim{1}, Dim{6}}}));
}

TEST_F(ModelEditorTest, single_input_single_output_double_resizing)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{4}, Dim{6}}};
  editor.resize_inputs(std::vector<Shape>{Shape{Dim{6}, Dim{8}}}).resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(), (std::vector<Shape>{Shape{Dim{4}, Dim{1}, Dim{6}}}));
}
