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

TEST_F(GenModelTest, OneOp_BroadcastTo_1D_to_2D)
{
  CircleGen cgen;
  const uint32_t shape_buf = cgen.addBuffer(std::vector<int32_t>{3, 3});
  int shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, shape_buf});
  int in = cgen.addTensor({{3}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{3, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorBroadcastTo({{in, shape}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3}}, {{1, 2, 3, 1, 2, 3, 1, 2, 3}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_BroadcastTo_2D_to_3D)
{
  CircleGen cgen;
  const uint32_t shape_buf = cgen.addBuffer(std::vector<int32_t>{3, 2, 2});
  int shape = cgen.addTensor({{3}, circle::TensorType::TensorType_INT32, shape_buf});
  int in = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorBroadcastTo({{in, shape}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_BroadcastTo_3D_to_3D)
{
  CircleGen cgen;
  const uint32_t shape_buf = cgen.addBuffer(std::vector<int32_t>{2, 3, 2});
  int shape = cgen.addTensor({{3}, circle::TensorType::TensorType_INT32, shape_buf});
  int in = cgen.addTensor({{2, 1, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorBroadcastTo({{in, shape}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_BroadcastTo_InputOutputDifferentType)
{
  CircleGen cgen;
  const uint32_t shape_buf = cgen.addBuffer(std::vector<int32_t>{3, 3});
  int shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, shape_buf});
  int in = cgen.addTensor({{3}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{3, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorBroadcastTo({{in, shape}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3}}, {{1, 2, 3, 1, 2, 3, 1, 2, 3}}));
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_BroadcastTo_1D_to_2D_InvalidShape)
{
  CircleGen cgen;
  const uint32_t shape_buf = cgen.addBuffer(std::vector<int32_t>{3, 2});
  int shape = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, shape_buf});
  int in = cgen.addTensor({{3}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorBroadcastTo({{in, shape}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3}}, {{1, 2, 3, 1, 2, 3}}));
  _context->setBackends({"cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_BroadcastTo_2D_to_3D_InvalidShape)
{
  CircleGen cgen;
  const uint32_t shape_buf = cgen.addBuffer(std::vector<int32_t>{2, 1, 3});
  int shape = cgen.addTensor({{3}, circle::TensorType::TensorType_INT32, shape_buf});
  int in = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2, 1, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorBroadcastTo({{in, shape}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 1, 2, 3}}, {{1, 2, 3, 1, 2, 3}}));
  _context->setBackends({"cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_BroadcastTo_3D_to_3D_InvalidShape)
{
  CircleGen cgen;
  const uint32_t shape_buf = cgen.addBuffer(std::vector<int32_t>{2, 3, 2});
  int shape = cgen.addTensor({{3}, circle::TensorType::TensorType_INT32, shape_buf});
  int in = cgen.addTensor({{2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorBroadcastTo({{in, shape}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 2, 1, 2, 1, 2, 1, 2}}, {{1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2}}));
  _context->setBackends({"cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_BroadcastTo_InvalidShapeType)
{
  CircleGen cgen;
  const uint32_t shape_buf = cgen.addBuffer(std::vector<float>{3, 3});
  int shape = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32, shape_buf});
  int in = cgen.addTensor({{3}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{3, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorBroadcastTo({{in, shape}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3}}, {{1, 2, 3, 1, 2, 3, 1, 2, 3}}));
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
