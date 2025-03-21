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
  auto model_data = std::make_shared<ModelData>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(model_data);
  const auto new_input_shapes = Shapes{Shape{Dim{4}, Dim{6}}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(model_data->input_shapes(), new_input_shapes);
  EXPECT_EQ(model_data->output_shapes(), (Shapes{Shape{Dim{4}, Dim{1}, Dim{6}}}));
}

TEST_F(ModelEditorTest, single_input_two_outputs)
{
  auto model_data = std::make_shared<ModelData>(_test_models_dir + "/CSE_Quantize_000.circle");
  ModelEditor editor(model_data);
  const auto new_input_shapes = Shapes{Shape{Dim{1}, Dim{6}, Dim{6}, Dim{4}}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(model_data->input_shapes(), new_input_shapes);
  EXPECT_EQ(model_data->output_shapes(),
            (Shapes{Shape{Dim{1}, Dim{6}, Dim{6}, Dim{4}}, Shape{Dim{1}, Dim{6}, Dim{6}, Dim{4}}}));
}

TEST_F(ModelEditorTest, two_inputs_single_output)
{
  auto model_data = std::make_shared<ModelData>(_test_models_dir + "/Add_000.circle");
  ModelEditor editor(model_data);
  const auto new_input_shapes =
    Shapes{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}}, Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(model_data->input_shapes(), new_input_shapes);
  EXPECT_EQ(model_data->output_shapes(), (Shapes{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}}}));
}

TEST_F(ModelEditorTest, two_inputs_two_outputs)
{
  auto model_data =
    std::make_shared<ModelData>(_test_models_dir + "/Part_Add_Sqrt_Rsqrt_000.circle");
  ModelEditor editor(model_data);
  const auto new_input_shapes =
    Shapes{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}}, Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}}};
  editor.resize_inputs(new_input_shapes);
  EXPECT_EQ(model_data->input_shapes(), new_input_shapes);
  EXPECT_EQ(model_data->output_shapes(),
            (Shapes{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}}, Shape{Dim{1}, Dim{5}, Dim{5}, Dim{2}}}));
}

TEST_F(ModelEditorTest, neg_not_all_input_shapes_provided)
{
  auto model_data = std::make_shared<ModelData>(_test_models_dir + "/Add_000.circle");
  ModelEditor editor(model_data);
  try
  {
    editor.resize_inputs(Shapes{Shape{Dim{1}, Dim{5}, Dim{5}, Dim{3}}});
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
  auto model_data = std::make_shared<ModelData>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(model_data);
  try
  {
    editor.resize_inputs(Shapes{Shape{Dim{3}}});
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

TEST_F(ModelEditorTest, neg_shape_inference_failed)
{
  auto model_data = std::make_shared<ModelData>(_test_models_dir + "/DepthwiseConv2D_000.circle");
  ModelEditor editor(model_data);
  EXPECT_THROW(editor.resize_inputs(Shapes{Shape{Dim{1}, Dim{64}, Dim{64}, Dim{8}},
                                           Shape{Dim{1}, Dim{2}, Dim{2}, Dim{3}}}),
               oops::UserExn);
}

TEST_F(ModelEditorTest, save_without_change)
{
  auto model_data = std::make_shared<ModelData>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(model_data);
  std::stringstream out_stream;
  model_data->save(out_stream);
  const std::string &model_buf_str = out_stream.str();
  std::vector<uint8_t> model_buffer(std::begin(model_buf_str), std::end(model_buf_str));
  model_buffer.insert(std::end(model_buffer), std::begin(model_buf_str), std::end(model_buf_str));
  auto model_data_2 = std::make_shared<ModelData>(model_buffer);
  ModelEditor editor_2(model_data_2);
  EXPECT_EQ(model_data_2->input_shapes(), (Shapes{Shape{Dim{3}, Dim{3}}}));
  EXPECT_EQ(model_data_2->output_shapes(), (Shapes{Shape{Dim{3}, Dim{1}, Dim{3}}}));
}

TEST_F(ModelEditorTest, save_after_resizing)
{
  auto model_data = std::make_shared<ModelData>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(model_data);
  std::stringstream out_stream;
  const auto new_input_shapes = Shapes{Shape{Dim{4}, Dim{6}}};
  editor.resize_inputs(new_input_shapes);
  model_data->save(out_stream);
  const std::string &model_buf_str = out_stream.str();
  std::vector<uint8_t> model_buffer(std::begin(model_buf_str), std::end(model_buf_str));
  model_buffer.insert(std::end(model_buffer), std::begin(model_buf_str), std::end(model_buf_str));
  auto model_data_2 = std::make_shared<ModelData>(model_buffer);
  ModelEditor editor_2(model_data_2);
  EXPECT_EQ(model_data_2->input_shapes(), new_input_shapes);
  EXPECT_EQ(model_data_2->output_shapes(), (Shapes{Shape{Dim{4}, Dim{1}, Dim{6}}}));
}

TEST_F(ModelEditorTest, single_input_single_output_double_resizing)
{
  auto model_data = std::make_shared<ModelData>(_test_models_dir + "/ExpandDims_000.circle");
  ModelEditor editor(model_data);
  const auto new_input_shapes = Shapes{Shape{Dim{4}, Dim{6}}};
  editor.resize_inputs(Shapes{Shape{Dim{6}, Dim{8}}}).resize_inputs(new_input_shapes);
  EXPECT_EQ(model_data->input_shapes(), new_input_shapes);
  EXPECT_EQ(model_data->output_shapes(), (Shapes{Shape{Dim{4}, Dim{1}, Dim{6}}}));
}
