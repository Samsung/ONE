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

#include "GenModelTest.h"

// beta = 0.1
// input/output shape: {1, 2, 1, 4}
struct SoftmaxParam
{
  TestCaseData tcd;
  circle::TensorType data_type = circle::TensorType::TensorType_FLOAT32;
  float input_scale = 0.0f;
  int64_t input_zero_point = 0;
};

class SoftmaxVariation : public GenModelTest, public ::testing::WithParamInterface<SoftmaxParam>
{
};

// Test with different value type
INSTANTIATE_TEST_SUITE_P(
  GenModelTest, SoftmaxVariation,
  ::testing::Values(
    // float value
    SoftmaxParam{
      uniformTCD<float>({{0, -6, 2, 4, 3, -2, 10, 1}},
                        {{.23463, .12877, .28658, .35003, .22528, .13664, .45365, .18443}})},
    // uint8 value
    SoftmaxParam{
      uniformTCD<uint8_t>({{10, 4, 12, 14, 13, 8, 20, 11}}, {{60, 33, 73, 90, 58, 35, 116, 47}}),
      circle::TensorType::TensorType_UINT8, 1.0, 10},
    // int8 value
    SoftmaxParam{
      uniformTCD<int8_t>({{0, -6, 2, 4, 3, -2, 10, 1}}, {{-68, -95, -55, -38, -70, -93, -12, -81}}),
      circle::TensorType::TensorType_INT8, 1.0, 0}));

TEST_P(SoftmaxVariation, Test)
{
  auto &param = GetParam();

  CircleGen cgen;

  // NNAPI spec and tflite test use fixed output scale and zero-point
  float out_scale = 0.0;
  int64_t out_zero_point = 0;
  if (param.data_type == circle::TensorType::TensorType_UINT8)
  {
    out_scale = 1.0f / 256;
  }
  else if (param.data_type == circle::TensorType::TensorType_INT8)
  {
    out_scale = 1.0f / 256;
    out_zero_point = -128;
  }

  int input =
    cgen.addTensor({{1, 2, 1, 4}, param.data_type}, param.input_scale, param.input_zero_point);
  int out = cgen.addTensor({{1, 2, 1, 4}, param.data_type}, out_scale, out_zero_point);
  cgen.addOperatorSoftmax({{input}, {out}}, 0.1);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(param.tcd);
  _context->setBackends({"cpu", "acl_neon", "acl_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Softmax_Invaild_Beta)
{
  CircleGen cgen;
  int input = cgen.addTensor({{4, 1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{4, 1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorSoftmax({{input}, {out}}, 0.1);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{-1., 0., 1., 1.}}, {{-1., -1., -1., -1.}}));
  _context->setBackends({"gpu_cl"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Softmax)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorSoftmax({{lhs}, {out}}, 1.0);
  cgen.setInputsAndOutputs({lhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>(
    {{-1., 0., 1., 1.}},
    {{0.054064586758613586, 0.14696279168128967, 0.39948627352714539, 0.39948627352714539}}));
  _context->setBackends({"acl_cl", "cpu", "gpu_cl"});

  SUCCEED();
}

TEST_P(SoftmaxVariation, neg_Type)
{
  auto &param = GetParam();

  CircleGen cgen;
  int input =
    cgen.addTensor({{1, 2, 1, 4}, param.data_type}, param.input_scale, param.input_zero_point);
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_BOOL});
  cgen.addOperatorSoftmax({{input}, {out}}, 0.1);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
