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
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

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
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

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
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, rhs_buf});
  int out = cgen.addTensor({{1, 2, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});

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
