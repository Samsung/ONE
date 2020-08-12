/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <nnfw_internal.h>

#include <fstream>

#include "CircleGen.h"
#include "fixtures.h"

/**
 * @brief Generated Model test fixture for a one time inference
 *
 * This fixture is for one-time inference test with variety of generated models.
 * It is the user's responsiblity to create @c _cbuf , @c _ref_inputs and @c _ref_outputs in the
 * test body, which are generated circle buffer, model input data and output data respectively.
 * The rest(calling API functions for execution) is done by @c Setup and @c TearDown .
 *
 */
class GenModelTest : public ::testing::Test
{
protected:
  void SetUp() override { NNFW_ENSURE_SUCCESS(nnfw_create_session(&_so.session)); }

  void TearDown() override
  {
    NNFW_ENSURE_SUCCESS(nnfw_load_circle_from_buffer(_so.session, _cbuf.buffer(), _cbuf.size()));
    NNFW_ENSURE_SUCCESS(nnfw_prepare(_so.session));

    // In/Out buffer settings
    {
      uint32_t num_inputs;
      NNFW_ENSURE_SUCCESS(nnfw_input_size(_so.session, &num_inputs));
      _so.inputs.resize(num_inputs);
      for (uint32_t ind = 0; ind < _so.inputs.size(); ind++)
      {
        nnfw_tensorinfo ti;
        NNFW_ENSURE_SUCCESS(nnfw_input_tensorinfo(_so.session, ind, &ti));
        uint64_t input_elements = num_elems(&ti);
        _so.inputs[ind].resize(input_elements);

        ASSERT_EQ(nnfw_set_input(_so.session, ind, ti.dtype, _so.inputs[ind].data(),
                                 sizeof(float) * input_elements),
                  NNFW_STATUS_NO_ERROR);
      }

      uint32_t num_outputs;
      NNFW_ENSURE_SUCCESS(nnfw_output_size(_so.session, &num_outputs));
      _so.outputs.resize(num_outputs);
      for (uint32_t ind = 0; ind < _so.outputs.size(); ind++)
      {
        nnfw_tensorinfo ti;
        NNFW_ENSURE_SUCCESS(nnfw_output_tensorinfo(_so.session, ind, &ti));
        uint64_t output_elements = num_elems(&ti);
        _so.outputs[ind].resize(output_elements);
        ASSERT_EQ(nnfw_set_output(_so.session, ind, ti.dtype, _so.outputs[ind].data(),
                                  sizeof(float) * output_elements),
                  NNFW_STATUS_NO_ERROR);
      }
    }

    // Set input values, run, and check output values
    {
      ASSERT_EQ(_so.inputs.size(), _ref_inputs.size());
      for (uint32_t i = 0; i < _so.inputs.size(); i++)
      {
        // Fill the values
        ASSERT_EQ(_so.inputs[i].size(), _ref_inputs[i].size());
        memcpy(_so.inputs[i].data(), _ref_inputs[i].data(), _so.inputs[i].size() * sizeof(float));
      }

      NNFW_ENSURE_SUCCESS(nnfw_run(_so.session));

      ASSERT_EQ(_so.outputs.size(), _ref_outputs.size());
      for (uint32_t i = 0; i < _so.outputs.size(); i++)
      {
        // Check output tensor values
        auto &ref_output = _ref_outputs[i];
        auto &output = _so.outputs[i];
        ASSERT_EQ(output.size(), ref_output.size());
        for (uint32_t e = 0; e < ref_output.size(); e++)
          EXPECT_NEAR(ref_output[e], output[e], 0.001); // TODO better way for handling FP error?
      }
    }

    NNFW_ENSURE_SUCCESS(nnfw_close_session(_so.session));
  }

protected:
  SessionObject _so;
  CircleBuffer _cbuf;
  std::vector<std::vector<float>> _ref_inputs;
  std::vector<std::vector<float>> _ref_outputs;
};

TEST_F(GenModelTest, OneOp_Add_VarToConst)
{
  CircleGen cgen;
  std::vector<float> rhs_data{5, 4, 7, 4};
  uint32_t rhs_buf = cgen.addBuffer(rhs_data);
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, rhs_buf});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs}, {out});
  _cbuf = cgen.finish();

  _ref_inputs = {{1, 3, 2, 4}};
  _ref_outputs = {{6, 7, 9, 8}};
}

TEST_F(GenModelTest, OneOp_Add_VarToVar)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});
  _cbuf = cgen.finish();

  _ref_inputs = {{1, 3, 2, 4}, {5, 4, 7, 4}};
  _ref_outputs = {{6, 7, 9, 8}};
}

TEST_F(GenModelTest, OneOp_AvgPool2D)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 2, 2, 2, 2,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});
  _cbuf = cgen.finish();

  _ref_inputs = {{1, 3, 2, 4}};
  _ref_outputs = {{2.5}};
}

TEST_F(GenModelTest, OneOp_Cos)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorCos({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});
  _cbuf = cgen.finish();

  const float pi = 3.141592653589793;
  _ref_inputs = {{0, pi / 2, pi, 7}};
  _ref_outputs = {{1, 0, -1, 0.75390225434}};
}
