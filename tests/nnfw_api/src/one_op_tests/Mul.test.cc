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

#include <memory>

TEST_F(GenModelTest, OneOp_Mul_Uint8_VarVar)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 1.0, 3);
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 2.0, 1);
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 0.5, 2);
  cgen.addOperatorMul({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<uint8_t>({{3, 12, 5, 2}, {5, 4, 7, 0}}, {{2, 110, 50, 6}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Mul_Int8_VarVar)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT8}, 1.0, 2);
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT8}, 2.0, 3);
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT8}, 0.5, -6);
  cgen.addOperatorMul({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<int8_t>({{1, 3, 2, 4}, {5, -4, -7, 4}}, {{-14, -34, -6, 2}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_MulBroadcast_Uint8_VarVar)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 1.0, 3);
  int rhs = cgen.addTensor({{1, 1, 1, 1}, circle::TensorType::TensorType_UINT8}, 2.0, 1);
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 0.5, 2);
  cgen.addOperatorMul({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<uint8_t>({{3, 12, 5, 4}, {5}}, {{2, 146, 34, 18}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_MulBroadcast_Int8_VarVar)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT8}, 1.0, 2);
  int rhs = cgen.addTensor({{1, 1, 1, 1}, circle::TensorType::TensorType_INT8}, 2.0, 3);
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT8}, 0.5, -6);
  cgen.addOperatorMul({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<int8_t>({{1, 3, 2, 4}, {5}}, {{-14, 2, -6, 10}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_MulBroadcast_Bool_VarVar)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_BOOL});
  int rhs = cgen.addTensor({{1, 1, 2, 1}, circle::TensorType::TensorType_BOOL});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_BOOL});
  cgen.addOperatorMul({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<uint8_t>({{true, false, false, true}, {true, false}},
                                            {{true, false, false, false}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Mul_InvalidType)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 0.1, 2);
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorMul({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Mul_InvalidShape)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorMul({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Mul_OneOperand)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorMul({{in}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Mul_ThreeOperands)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorMul({{in, in, in}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Mul_Invalid_Broadcast_Bool)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 1, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorMul({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}
