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
  _context->addTestCase(TestCaseData{}.addInput<float>({1, 4, 2, 3}).addOutput<int32_t>({1, 0}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_ArgMin_AxisToConst)
{
  CircleGen cgen;
  const auto output_type = circle::TensorType::TensorType_INT32;
  std::vector<int32_t> axis_data{1};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 1}, output_type});
  cgen.addOperatorArgMin({{in, axis}, {out}}, output_type);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}.addInput<float>({1, 4, 2, 3}).addOutput<int32_t>({0, 1}));
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
  _context->addTestCase(TestCaseData{}.addInput<float>({1, 4, 2, 3}).addOutput<int64_t>({1, 0}));
  _context->setBackends({"acl_cl", "cpu"});

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
  _context->addTestCase(TestCaseData{}
                            .addInput<float>({1, 4, 2, 3})
                            .addInput<int32_t>({-3})
                            .addOutput<int32_t>({1, 0}));
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
  _context->expectFailCompile();

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
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_ArgMax_InType)
{
  CircleGen cgen;
  const auto output_type = circle::TensorType::TensorType_INT32;
  std::vector<int32_t> axis_data{4};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_BOOL});
  int out = cgen.addTensor({{1, 2, 1}, output_type});
  cgen.addOperatorArgMax({{in, axis}, {out}}, output_type);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_ArgMax_AxisType)
{
  CircleGen cgen;
  const auto output_type = circle::TensorType::TensorType_FLOAT32;
  std::vector<float> axis_data{4};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, axis_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 1}, output_type});
  cgen.addOperatorArgMax({{in, axis}, {out}}, output_type);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_ArgMax_OutType)
{
  CircleGen cgen;
  const auto output_type = circle::TensorType::TensorType_FLOAT32;
  std::vector<int32_t> axis_data{4};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 1}, output_type});
  cgen.addOperatorArgMax({{in, axis}, {out}}, output_type);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_ArgMax_paramType)
{
  CircleGen cgen;
  const auto output_type = circle::TensorType::TensorType_INT32;
  std::vector<int32_t> axis_data{4};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 1}, output_type});
  cgen.addOperatorArgMax({{in, axis}, {out}}, circle::TensorType::TensorType_INT64);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
