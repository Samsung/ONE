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

TEST_F(GenModelTest, OneOp_Select)
{
  CircleGen cgen;
  std::vector<uint8_t> cond_data{1, 1, 0, 1};
  uint32_t cond_buf = cgen.addBuffer(cond_data);
  int cond = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_BOOL, cond_buf});
  int in_true = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int in_false = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorSelect({{cond, in_true, in_false}, {out}});
  cgen.setInputsAndOutputs({in_true, in_false}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0, 1, 2, 3}, {4, 5, 6, 7}}, {{0, 1, 6, 3}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_SelectV2_Broadcast)
{
  CircleGen cgen;
  std::vector<uint8_t> cond_data{1, 0};
  uint32_t cond_buf = cgen.addBuffer(cond_data);
  int cond = cgen.addTensor({{1, 2, 1, 1}, circle::TensorType::TensorType_BOOL, cond_buf});
  int in_true = cgen.addTensor({{1, 1, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int in_false = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorSelectV2({{cond, in_true, in_false}, {out}});
  cgen.setInputsAndOutputs({in_true, in_false}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0, 1}, {4, 5, 6, 7}}, {{0, 1, 6, 7}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Select_InputType)
{
  CircleGen cgen;
  std::vector<uint8_t> cond_data{1, 1, 0, 1};
  uint32_t cond_buf = cgen.addBuffer(cond_data);
  int cond = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_BOOL, cond_buf});
  int in_true = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int in_false = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorSelect({{cond, in_true, in_false}, {out}});
  cgen.setInputsAndOutputs({in_true, in_false}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Select_CondType)
{
  CircleGen cgen;
  std::vector<uint8_t> cond_data{1, 1, 0, 1};
  uint32_t cond_buf = cgen.addBuffer(cond_data);
  int cond = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_UINT8, cond_buf});
  int in_true = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int in_false = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorSelect({{cond, in_true, in_false}, {out}});
  cgen.setInputsAndOutputs({in_true, in_false}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
