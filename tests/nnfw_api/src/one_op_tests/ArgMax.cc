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

TEST_F(GenModelTest, OneOp_ArgMax_AxisToConst)
{
  CircleGen cgen;
  const auto output_type = circle::TensorType::TensorType_INT32;
  std::vector<int32_t> axis_data{1};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 1}, output_type});
  cgen.addOperatorArgMax({{in, axis}, {out}}, output_type);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  TestCaseData tcd;
  tcd.addInput(std::vector<float>{1, 4, 2, 3});
  tcd.addOutput(std::vector<int32_t>{1, 0});
  _context->addTestCase(tcd);
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_ArgMax_Int64_AxisToConst)
{
  CircleGen cgen;
  const auto output_type = circle::TensorType::TensorType_INT64;
  std::vector<int32_t> axis_data{1};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 1}, output_type});
  cgen.addOperatorArgMax({{in, axis}, {out}}, output_type);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  TestCaseData tcd;
  tcd.addInput(std::vector<float>{1, 4, 2, 3});
  tcd.addOutput(std::vector<int64_t>{1, 0});
  _context->addTestCase(tcd);
  _context->setBackends({"acl_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_ArgMax_AxisToVar)
{
  CircleGen cgen;
  const auto output_type = circle::TensorType::TensorType_INT32;
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 1}, output_type});
  cgen.addOperatorArgMax({{in, axis}, {out}}, output_type);
  cgen.setInputsAndOutputs({in, axis}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  TestCaseData tcd;
  tcd.addInput(std::vector<float>{1, 4, 2, 3});
  tcd.addInput(std::vector<int32_t>{-3});
  tcd.addOutput(std::vector<int32_t>{1, 0});
  _context->addTestCase(tcd);
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_ArgMax_InvalidAxis0)
{
  CircleGen cgen;
  const auto output_type = circle::TensorType::TensorType_INT32;
  std::vector<int32_t> axis_data{4};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 1}, output_type});
  cgen.addOperatorArgMax({{in, axis}, {out}}, output_type);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->setCompileFail();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_ArgMax_InvalidAxis1)
{
  CircleGen cgen;
  const auto output_type = circle::TensorType::TensorType_INT32;
  std::vector<int32_t> axis_data{-3};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int in = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2}, output_type});
  cgen.addOperatorArgMax({{in, axis}, {out}}, output_type);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->setCompileFail();

  SUCCEED();
}
