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

TEST_F(GenModelTest, OneOp_Tile_ConstMul)
{
  CircleGen cgen;
  std::vector<int32_t> mul_data{1, 2};
  uint32_t mul_buf = cgen.addBuffer(mul_data);
  int in = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  int mul = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, mul_buf});
  int out = cgen.addTensor({{2, 6}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTile({{in, mul}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
      uniformTCD<float>({{1, 2, 3, 4, 5, 6}}, {{1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

// Variable mul input is not supported yet
TEST_F(GenModelTest, DISABLED_OneOp_Tile_VarMul)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_INT32});
  int mul = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{2, 6}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorTile({{in, mul}, {out}});
  cgen.setInputsAndOutputs({in, mul}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  TestCaseData tcd;
  tcd.addInput(std::vector<float>{1, 2, 3, 4, 5, 6});
  tcd.addInput(std::vector<int32_t>{1, 2});
  tcd.addOutput(std::vector<float>{1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6});
  _context->addTestCase(tcd);
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Tile)
{
  CircleGen cgen;
  std::vector<int32_t> mul_data{1, 2, 1, 2};
  uint32_t mul_buf = cgen.addBuffer(mul_data);
  int in = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  // 2D multiples input is not supported
  int mul = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_INT32, mul_buf});
  int out = cgen.addTensor({{2, 6}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTile({{in, mul}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->setCompileFail();

  SUCCEED();
}
