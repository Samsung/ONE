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
  const auto new_input_shapes = std::vector<Shape>{Shape{4, 6}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(), (std::vector<Shape>{Shape{4, 1, 6}}));
}

TEST_F(ModelEditorTest, single_input_two_outputs)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/CSE_Quantize_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{1, 6, 6, 4}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(),
            (std::vector<Shape>{Shape{1, 6, 6, 4},
                                Shape{1, 6, 6, 4}}));
}

TEST_F(ModelEditorTest, two_inputs_single_output)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/Add_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{1, 5, 5, 3},
                                                   Shape{1, 5, 5, 3}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(),
            (std::vector<Shape>{Shape{1, 5, 5, 3}}));
}

TEST_F(ModelEditorTest, two_inputs_two_outputs)
{
  auto circle_model =
    std::make_shared<CircleModel>(_test_models_dir + "/Part_Add_Sqrt_Rsqrt_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{1, 5, 5, 2},
                                                   Shape{1, 5, 5, 2}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(),
            (std::vector<Shape>{Shape{1, 5, 5, 2},
                                Shape{1, 5, 5, 2}}));
}

TEST_F(ModelEditorTest, resize_applied_after_save)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  std::stringstream out_stream;
  const auto new_input_shapes = std::vector<Shape>{Shape{4, 6}};
  editor.resize_inputs(new_input_shapes);
  circle_model->save(out_stream);
  const std::string &model_buf_str = out_stream.str();
  std::vector<uint8_t> model_buffer(std::begin(model_buf_str), std::end(model_buf_str));
  model_buffer.insert(std::end(model_buffer), std::begin(model_buf_str), std::end(model_buf_str));

  auto circle_model_from_saved_buffer = std::make_shared<CircleModel>(model_buffer);
  EXPECT_EQ(circle_model_from_saved_buffer->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model_from_saved_buffer->output_shapes(), (std::vector<Shape>{Shape{4, 1, 6}}));
}

TEST_F(ModelEditorTest, single_input_single_output_double_resizing)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{4, 6}};
  editor.resize_inputs(std::vector<Shape>{Shape{6, 8}}).resize_inputs(new_input_shapes);
  // check if the last applied shape is set after double resizing call
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(), (std::vector<Shape>{Shape{4, 1, 6}}));
}

TEST_F(ModelEditorTest, change_input_rank)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{1, 2, 3, 4}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(), (std::vector<Shape>{Shape{1, 1, 2, 3, 4}}));
}

TEST_F(ModelEditorTest, resize_to_dynamic)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{Dim{4}, Dim::dynamic()}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(), (std::vector<Shape>{Shape{Dim{4}, Dim{1}, Dim::dynamic()}}));
}

TEST_F(ModelEditorTest, not_all_input_shapes_provided_NEG)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/Add_000.circle");
  ModelEditor editor(circle_model);
  try
  {
    editor.resize_inputs(std::vector<Shape>{Shape{1, 5, 5, 3}});
    FAIL() << "Unexpected successful resizing with invalid shapes.";
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Expected 2 shapes but provided only 1"));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}

TEST_F(ModelEditorTest, exception_during_shape_inference_NEG)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/Add_000.circle");
  ModelEditor editor(circle_model);
  try
  {
    editor.resize_inputs(std::vector<Shape>{Shape{1, 2, 3},
                                                    Shape{4, 5, 6}});
    FAIL() << "Unexpected successful resizing with invalid shapes.";
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Exception during shape inference with message:"));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}
