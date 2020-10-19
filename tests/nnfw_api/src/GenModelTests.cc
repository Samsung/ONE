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

/**
 * @file This file contains miscellaneous GenModelTest test cases.
 *
 */

#include "GenModelTest.h"

#include <memory>

TEST_F(GenModelTest, UnusedConstOutputOnly)
{
  // A single tensor which is constant
  CircleGen cgen;
  uint32_t const_buf = cgen.addBuffer(std::vector<float>{9, 8, 7, 6});
  int out_const = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, const_buf});
  cgen.setInputsAndOutputs({}, {out_const});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({}, {{9, 8, 7, 6}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, UnusedConstOutputAndAdd)
{
  // A single tensor which is constant + an Add op
  CircleGen cgen;
  uint32_t rhs_buf = cgen.addBuffer(std::vector<float>{5, 4, 7, 4});
  uint32_t const_buf = cgen.addBuffer(std::vector<float>{9, 8, 7, 6});
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, rhs_buf});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out_const = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, const_buf});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs}, {out, out_const});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 3, 2, 4}}, {{6, 7, 9, 8}, {9, 8, 7, 6}}));
  _context->addTestCase(uniformTCD<float>({{0, 1, 2, 3}}, {{5, 5, 9, 7}, {9, 8, 7, 6}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, UsedConstOutput)
{
  // (( Input 1 )) ---------\
  //                         |=> [ Add ] -> (( Output 1 ))
  // (( Const Output 2 )) --<
  //                         |=> [ Add ] -> (( Output 0 ))
  // (( Input 0 )) ---------/
  CircleGen cgen;
  uint32_t rhs_buf = cgen.addBuffer(std::vector<float>{6, 4, 8, 1});
  int in0 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int in1 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out0 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out1 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int const_out2 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, rhs_buf});
  cgen.addOperatorAdd({{in0, const_out2}, {out0}}, circle::ActivationFunctionType_NONE);
  cgen.addOperatorAdd({{const_out2, in1}, {out1}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in0, in1}, {out0, out1, const_out2});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 1, 1, 1}, {-1, -1, -1, -1}},
                                          {{7, 5, 9, 2}, {5, 3, 7, 0}, {6, 4, 8, 1}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, TensorBothInputOutput)
{
  // A single tensor which is an input and an output at the same time
  CircleGen cgen;
  int t = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.setInputsAndOutputs({t}, {t});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 3, 2, 4}}, {{1, 3, 2, 4}}));
  _context->addTestCase(uniformTCD<float>({{100, 300, 200, 400}}, {{100, 300, 200, 400}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, TensorBothInputOutputCrossed)
{
  // Two tensors which are an input and an output at the same time
  // But the order of inputs and outputs is changed.
  CircleGen cgen;
  int t1 = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int t2 = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  cgen.setInputsAndOutputs({t1, t2}, {t2, t1});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1}, {2}}, {{2}, {1}}));
  _context->addTestCase(uniformTCD<float>({{100}, {200}}, {{200}, {100}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneTensor_TwoOutputs)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out, out}); // Same tensors are used twice as output

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 1}, {2, 2}}, {{3, 3}, {3, 3}}));
  _context->addTestCase(uniformTCD<float>({{2, 4}, {7, 4}}, {{9, 8}, {9, 8}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneTensor_ThreeOutputs)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out, out, out}); // Same tensors are used 3 times as output

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1}, {2}}, {{3}, {3}, {3}}));
  _context->addTestCase(uniformTCD<float>({{2}, {7}}, {{9}, {9}, {9}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneTensor_InputAndTwoOutputs)
{
  CircleGen cgen;
  int t = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  cgen.setInputsAndOutputs({t}, {t, t}); // Same tensor is an input and 2 outputs

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 1}}, {{1, 1}, {1, 1}}));
  _context->addTestCase(uniformTCD<float>({{2, 4}}, {{2, 4}, {2, 4}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneTensor_InputAndTwoOutputsUsed)
{
  CircleGen cgen;
  int t = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  int o = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorNeg({{t}, {o}});
  cgen.setInputsAndOutputs({t}, {t, t, o}); // Same tensor is an input and 2 outputs

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 1}}, {{1, 1}, {1, 1}, {-1, -1}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneTensor_ConstAndThreeOutputs)
{
  CircleGen cgen;
  uint32_t const_buf = cgen.addBuffer(std::vector<float>{2, 5});
  int t = cgen.addTensor({{2}, circle::TensorType_FLOAT32, const_buf});
  cgen.setInputsAndOutputs({}, {t, t, t}); // A const tensor is 3 outputs

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({}, {{2, 5}, {2, 5}, {2, 5}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}
