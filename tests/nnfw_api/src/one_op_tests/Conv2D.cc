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

TEST_F(GenModelTest, OneOp_Conv2D)
{
  CircleGen cgen;
  std::vector<float> weight_data{-2, 3, -5, 3, 4, 4, 0, 0, -4, -1, -4, -2, 0, 2, 0, -1, 4, 0};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{2, 3};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1,
                         circle::ActivationFunctionType_NONE, 1, 1);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>(
    {{4, 0, -5, 1, 0, 4, -1, 1, -1, -3, 3, -2, -4, 1, -2, 2, 4, -4, 2, 2, 0, 4, -1, -2, 4}},
    {{47, -4, -25, 9, 10, 10, -13, 11, -14, -26, -12, 26, 20, 40, 1, 3, 11, 4}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "ruy", "xnnpack"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Conv2D_Stride)
{
  CircleGen cgen;
  std::vector<float> weight_data{-2, 3, -5, 3, 4, 4, 0, 0, -4, -1, -4, -2, 0, 2, 0, -1, 4, 0};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{2, 3};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorConv2D({{in, weight, bias}, {out}}, circle::Padding_SAME, 2, 2,
                         circle::ActivationFunctionType_NONE, 1, 1);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>(
    {{4, 0, -5, 1, 0, 4, -1, 1, -1, -3, 3, -2, -4, 1, -2, 2, 4, -4, 2, 2, 0, 4, -1, -2, 4}},
    {{22, 27, -10, -2, 5, -8, 7, 3, -14, -26, -10, 18, 4, -13, -28, 9, 14, 1}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "ruy", "xnnpack"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Conv2D_Dilation)
{
  CircleGen cgen;
  std::vector<float> weight_data{-2, 3, -5, 3, 4, 4, 0, 0, -4, -1, -4, -2, 0, 2, 0, -1, 4, 0};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{2, 3};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1,
                         circle::ActivationFunctionType_NONE, 2, 2);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>(
    {{4, 0, -5, 1, 0, 4, -1, 1, -1, -3, 3, -2, -4, 1, -2, 2, 4, -4, 2, 2, 0, 4, -1, -2, 4}},
    {{-52, 7}}));
  _context->setBackends({"cpu", "ruy", "xnnpack"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Conv2D_Type)
{
  CircleGen cgen;
  std::vector<float> weight_data{-2, 3, -5, 3, 4, 4, 0, 0, -4, -1, -4, -2, 0, 2, 0, -1, 4, 0};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{2, 3};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT16});
  cgen.addOperatorConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1,
                         circle::ActivationFunctionType_NONE, 1, 1);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Conv2D_Stride)
{
  CircleGen cgen;
  std::vector<float> weight_data{-2, 3, -5, 3, 4, 4, 0, 0, -4, -1, -4, -2, 0, 2, 0, -1, 4, 0};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{2, 3};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorConv2D({{in, weight, bias}, {out}}, circle::Padding_SAME, 0, 0,
                         circle::ActivationFunctionType_NONE, 1, 1);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Conv2D_Dilation)
{
  CircleGen cgen;
  std::vector<float> weight_data{-2, 3, -5, 3, 4, 4, 0, 0, -4, -1, -4, -2, 0, 2, 0, -1, 4, 0};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{2, 3};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 5, 5, 1}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{2, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1,
                         circle::ActivationFunctionType_NONE, 0, 0);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
