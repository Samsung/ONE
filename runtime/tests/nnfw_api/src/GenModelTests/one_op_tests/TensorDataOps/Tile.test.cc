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

TEST_F(GenModelTest, OneOp_Tile_MulToConst)
{
  CircleGen cgen;
  std::vector<int32_t> multiplies_data{2, 3, 1};
  uint32_t multiplies_buf = cgen.addBuffer(multiplies_data);
  int multiplies = cgen.addTensor({{3}, circle::TensorType::TensorType_INT32, multiplies_buf});
  int in = cgen.addTensor({{1, 2, 3}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2, 6, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTile({{in, multiplies}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{11, 12, 13, 21, 22, 23}},
                      {{11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23,
                        11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Tile_MulToVar)
{
  CircleGen cgen;
  int multiplies = cgen.addTensor({{3}, circle::TensorType::TensorType_INT32});
  int in = cgen.addTensor({{1, 2, 3}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2, 6, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTile({{in, multiplies}, {out}});
  cgen.setInputsAndOutputs({in, multiplies}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}
      .addInput<float>({11, 12, 13, 21, 22, 23})
      .addInput<int32_t>({2, 3, 1})
      .addOutput<float>({11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23,
                         11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23, 11, 12, 13, 21, 22, 23}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Tile_VarMul)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  int mul = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{2, 6}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTile({{in, mul}, {out}});
  cgen.setInputsAndOutputs({in, mul}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<float>({1, 2, 3, 4, 5, 6})
                          .addInput<int32_t>({1, 2})
                          .addOutput<float>({1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6}));
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
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Tile_InvalidMulSize)
{
  CircleGen cgen;
  std::vector<int32_t> multiplies_data{2, 6};
  uint32_t multiplies_buf = cgen.addBuffer(multiplies_data);
  int multiplies = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, multiplies_buf});
  int in = cgen.addTensor({{1, 2, 3}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2, 6, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorTile({{in, multiplies}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailCompile();

  SUCCEED();
}
