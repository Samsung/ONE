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

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 3, 2, 4}}, {{6, 7, 9, 8}}));
  _context->addTestCase(uniformTCD<float>({{0, 1, 2, 3}}, {{5, 5, 9, 7}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "gpu_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Add_VarToVar)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 3, 2, 4}, {5, 4, 7, 4}}, {{6, 7, 9, 8}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "gpu_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Add_VarToVarUint8)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 0.1, 1);
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 0.1, 2);
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 0.1, 4);
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<uint8_t>({{1, 3, 2, 4}, {5, 4, 7, 4}}, {{7, 8, 10, 9}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Add_VarToVarInt8)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT8}, 1., 2);
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT8}, 2., 3);
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT8}, 0.5, -6);
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<int8_t>({{1, 3, 2, 4}, {5, -4, -7, 4}}, {{0, -32, -46, 2}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_BroadcastAdd_VarToVarInt8)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT8}, 1., 2);
  int rhs = cgen.addTensor({{1, 1, 1, 1}, circle::TensorType::TensorType_INT8}, 2., 3);
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT8}, 0.5, -6);
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<int8_t>({{1, 3, 2, 4}, {5}}, {{0, 4, 2, 6}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Add_VarToVarSame)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{in, in}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 3, 2, 4}}, {{2, 6, 4, 8}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "gpu_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Add_VarToVarSize0)
{
  CircleGen cgen;
  int a = cgen.addTensor({{0}, circle::TensorType::TensorType_FLOAT32});
  int b = cgen.addTensor({{0}, circle::TensorType::TensorType_FLOAT32});
  int c = cgen.addTensor({{0}, circle::TensorType::TensorType_FLOAT32});
  int m = cgen.addTensor({{0}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{0}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{a, b}, {m}}, circle::ActivationFunctionType_NONE);
  cgen.addOperatorAdd({{m, c}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({a, b, c}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{}, {}, {}}, {{}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Add_InvalidType)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 0.1, 2);
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Add_DifferentQuant8Type)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT8}, 0.2, -3);
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8}, 0.1, 2);
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_INT8});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Add_InvalidShape)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Add_InvalidShapeConst)
{
  CircleGen cgen;
  std::vector<float> rhs_data{5, 4, 0, 7, 4, 0};
  uint32_t rhs_buf = cgen.addBuffer(rhs_data);
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32, rhs_buf});
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Add_OneOperand)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{in}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Add_ThreeOperands)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{in, in, in}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Add_NoOutput)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{in}, {}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Add_InvalidActivation)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}},
                      static_cast<circle::ActivationFunctionType>(128) /* Invalid value*/);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 3, 2, 4}, {5, 4, 7, 4}}, {{6, 7, 9, 8}}));
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Add_VarToVarSize0_InvalidShape)
{
  CircleGen cgen;
  int a = cgen.addTensor({{0}, circle::TensorType::TensorType_FLOAT32});
  int b = cgen.addTensor({{0}, circle::TensorType::TensorType_FLOAT32});
  int c = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  int m = cgen.addTensor({{0}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{0}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{a, b}, {m}}, circle::ActivationFunctionType_NONE);
  cgen.addOperatorAdd({{m, c}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({a, b, c}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailCompile();
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Add_VarToVarInt16)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT16}, 1., 2);
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT16}, 2., 3);
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT16}, 0.5, -6);
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  // _context->addTestCase(uniformTCD<int8_t>({{1, 3, 2, 4}, {5, -4, -7, 4}}, {{0, -32, -46, 2}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailCompile();

  SUCCEED();
}
