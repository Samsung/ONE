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

TEST_F(GenModelTest, OneOp_DepthwiseConv2D)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{1, 2, 3, 4};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12}},
                                          {{71, -34, 99, -20, 91, -26, 127, -4}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_DepthwiseConv2D_Dilation)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{0, 0, 0, 0};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 4, 4, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE, 2, 2);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                                              0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          }},
                                          {{13, 14, 0, 0, 0, 0, 11, 12, 5, 6, 0, 0, 0, 0, 3, 4}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_DepthwiseConv2D_Dilation_N_Stride)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{0, 0, 0, 0};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 6, 6, 1}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 1}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_SAME, 2, 2, 1,
                                  circle::ActivationFunctionType_NONE, 3, 3);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                                            0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                                          {{4, 0, 3, 0, 0, 0, 2, 0, 1}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_DepthwiseConv2D_Stride)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{1, 2, 3, 4};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 0, 0, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_DepthwiseConv2D_Dilation)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{0, 0, 0, 0};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 4, 4, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE, 0, 0);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_DepthwiseConv2D_Type)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{1, 2, 3, 4};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_UINT8});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
