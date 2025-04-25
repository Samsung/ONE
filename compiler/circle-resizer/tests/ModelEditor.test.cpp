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
    if (nullptr == path)
    {
      throw std::runtime_error("environmental variable ARTIFACTS_PATH required for circle-resizer "
                               "tests was not provided");
    }
    _test_models_dir = path;
  }

protected:
  std::string _test_models_dir;
};

TEST_F(ModelEditorTest, basic_tests)
{
  // single input, single output
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  auto new_input_shapes = std::vector<Shape>{Shape{4, 6}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(), (std::vector<Shape>{Shape{4, 1, 6}}));

  // single input, two outputs
  circle_model = std::make_shared<CircleModel>(_test_models_dir + "/CSE_Quantize_000.circle");
  editor = ModelEditor(circle_model);
  new_input_shapes = std::vector<Shape>{Shape{1, 6, 6, 4}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(),
            (std::vector<Shape>{Shape{1, 6, 6, 4}, Shape{1, 6, 6, 4}}));

  // two inputs, single output
  circle_model = std::make_shared<CircleModel>(_test_models_dir + "/Add_000.circle");
  editor = ModelEditor(circle_model);
  new_input_shapes = std::vector<Shape>{Shape{1, 5, 5, 3}, Shape{1, 5, 5, 3}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(), (std::vector<Shape>{Shape{1, 5, 5, 3}}));

  // two inputs two outputs
  circle_model =
    std::make_shared<CircleModel>(_test_models_dir + "/Part_Add_Sqrt_Rsqrt_000.circle");
  editor = ModelEditor(circle_model);
  new_input_shapes = std::vector<Shape>{Shape{1, 5, 5, 2}, Shape{1, 5, 5, 2}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(),
            (std::vector<Shape>{Shape{1, 5, 5, 2}, Shape{1, 5, 5, 2}}));

  // change even the input rank
  circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  editor = ModelEditor(circle_model);
  new_input_shapes = std::vector<Shape>{Shape{1, 2, 3, 4}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(), (std::vector<Shape>{Shape{1, 1, 2, 3, 4}}));
}

TEST_F(ModelEditorTest, special_cases)
{
  // resize to dynamic shape
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  auto new_input_shapes = std::vector<Shape>{Shape{Dim{4}, Dim::dynamic()}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(),
            (std::vector<Shape>{Shape{Dim{4}, Dim{1}, Dim::dynamic()}}));

  // resize to scalars
  circle_model = std::make_shared<CircleModel>(_test_models_dir + "/Add_000.circle");
  editor = ModelEditor(circle_model);
  new_input_shapes = std::vector<Shape>{Shape::scalar(), Shape::scalar()};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(), (std::vector<Shape>{Shape::scalar()}));
}

TEST_F(ModelEditorTest, resizing_applied_after_save)
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

TEST_F(ModelEditorTest, double_resizing)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(circle_model);
  const auto new_input_shapes = std::vector<Shape>{Shape{4, 6}};
  editor.resize_inputs(std::vector<Shape>{Shape{6, 8}}).resize_inputs(new_input_shapes);
  // check if the last applied shape is set after double resizing call
  EXPECT_EQ(circle_model->input_shapes(), new_input_shapes);
  EXPECT_EQ(circle_model->output_shapes(), (std::vector<Shape>{Shape{4, 1, 6}}));
}

TEST_F(ModelEditorTest, no_inputs_shapes_provided_NEG)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/Add_000.circle");
  ModelEditor editor(circle_model);
  try
  {
    editor.resize_inputs({});
    FAIL() << "Unexpected successful resizing with invalid shapes.";
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Expected 2 shapes but provided 0"));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}

TEST_F(ModelEditorTest, not_all_inputs_shapes_provided_NEG)
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
    EXPECT_THAT(err.what(), HasSubstr("Expected 2 shapes but provided 1"));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}

TEST_F(ModelEditorTest, to_much_inputs_shapes_provided_NEG)
{
  auto circle_model = std::make_shared<CircleModel>(_test_models_dir + "/Add_000.circle");
  ModelEditor editor(circle_model);
  try
  {
    editor.resize_inputs(std::vector<Shape>{Shape{1, 2}, Shape{3, 4}, Shape{5, 6}});
    FAIL() << "Unexpected successful resizing with invalid shapes.";
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Expected 2 shapes but provided 3"));
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
    editor.resize_inputs(std::vector<Shape>{Shape{1, 2, 3}, Shape{4, 5, 6}});
    FAIL() << "Unexpected successful resizing with invalid shapes.";
  }
  catch (const std::runtime_error &err)
  {
    EXPECT_THAT(err.what(), HasSubstr("Exception during resizing with message:"));
  }
  catch (...)
  {
    FAIL() << "Expected std::runtime_error, other exception thrown";
  }
}
