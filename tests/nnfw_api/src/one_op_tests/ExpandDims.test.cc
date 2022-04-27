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

TEST_F(GenModelTest, OneOp_ExpandDims)
{
  CircleGen cgen;

  std::vector<int32_t> axis_data{1};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int in = cgen.addTensor({{1, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int out = cgen.addTensor({{1, 1, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorExpandDims({{in, axis}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<float>({0.1, 0.3, 0.5, 0.7}).addOutput<float>({0.1, 0.3, 0.5, 0.7}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_ExpandDims_Int64AxisNeg)
{
  CircleGen cgen;

  std::vector<int64_t> axis_data{-1};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int in = cgen.addTensor({{1, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT64, axis_buf});
  int out = cgen.addTensor({{1, 4, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorExpandDims({{in, axis}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<float>({0.1, 0.3, 0.5, 0.7}).addOutput<float>({0.1, 0.3, 0.5, 0.7}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_neg_ExpandDims_Axis)
{
  CircleGen cgen;

  std::vector<int32_t> axis_data{4};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int in = cgen.addTensor({{1, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});
  int out = cgen.addTensor({{1, 1, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorExpandDims({{in, axis}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_neg_ExpandDims_AxisNegInput)
{
  CircleGen cgen;

  int in = cgen.addTensor({{1, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{1, 1, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorExpandDims({{in, axis}, {out}});
  cgen.setInputsAndOutputs({in, axis}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<float>({0.1, 0.3, 0.5, 0.7})
                          .addInput<int32_t>({-5})
                          .addOutput<float>({0.1, 0.3, 0.5, 0.7})
                          .expectFailRun());
  _context->setBackends({"cpu"});

  SUCCEED();
}
