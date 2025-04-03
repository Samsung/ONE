/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

TEST_F(GenModelTest, OneOp_BatchMatMul)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 3}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 3, 4}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorBatchMatMul({{lhs, rhs}, {out}}, false, false);
  cgen.setInputsAndOutputs({lhs, rhs}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<float>({1, 2, 3, 4, 5, 6})
                          .addInput<float>({7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})
                          .addOutput<float>({74, 80, 86, 92, 173, 188, 203, 218}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_BatchMatMul_InvalidType)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 3}, circle::TensorType::TensorType_INT32});
  int rhs = cgen.addTensor({{1, 3, 4}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{1, 2, 4}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorBatchMatMul({{lhs, rhs}, {out}}, false, false);
  cgen.setInputsAndOutputs({lhs}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_BatchMatMul_Const)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 3}, circle::TensorType::TensorType_FLOAT32});
  std::vector<float> const_data{7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  uint32_t rhs_buf = cgen.addBuffer(const_data);
  int rhs = cgen.addTensor({{1, 3, 4}, circle::TensorType::TensorType_FLOAT32, rhs_buf});
  int out = cgen.addTensor({{1, 2, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorBatchMatMul({{lhs, rhs}, {out}}, false, false);
  cgen.setInputsAndOutputs({lhs}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<float>({1, 2, 3, 4, 5, 6})
                          .addOutput<float>({74, 80, 86, 92, 173, 188, 203, 218}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_BatchMatMul_InvalidConst)
{
  // LHS constant is not allowed
  CircleGen cgen;
  std::vector<float> const_data{1, 2, 3, 4, 5, 6};
  uint32_t lhs_buf = cgen.addBuffer(const_data);
  int lhs = cgen.addTensor({{1, 2, 3}, circle::TensorType::TensorType_FLOAT32, lhs_buf});
  int rhs = cgen.addTensor({{1, 3, 4}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorBatchMatMul({{lhs, rhs}, {out}}, false, false);
  cgen.setInputsAndOutputs({lhs}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
