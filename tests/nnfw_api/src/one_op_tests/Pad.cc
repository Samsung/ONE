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

TEST_F(GenModelTest, OneOp_Pad)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> padding_data{0, 0, 1, 1, 1, 1, 0, 0};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{4, 2}, circle::TensorType::TensorType_INT32, padding_buf});
  int out = cgen.addTensor({{1, 4, 4, 1}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorPad({{in, padding}, {out}});
  cgen.setInputsAndOutputs({in}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase({{{1, 2, 3, 4}}, {{0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0}}});
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Pad_InvalidPadRank)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> padding_data{1, 1, 1, 1};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{4}, circle::TensorType::TensorType_INT32, padding_buf});
  int out = cgen.addTensor({{1, 4, 4, 1}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorPad({{in, padding}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->setCompileFail();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Pad_InvalidPadDim0)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> padding_data{1, 1, 1, 1};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_INT32, padding_buf});
  int out = cgen.addTensor({{1, 4, 4, 1}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorPad({{in, padding}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->setCompileFail();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Pad_InvalidPadDim1)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> padding_data{1, 1, 1, 1};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{4, 1}, circle::TensorType::TensorType_INT32, padding_buf});
  int out = cgen.addTensor({{2, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorPad({{in, padding}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->setCompileFail();

  SUCCEED();
}
