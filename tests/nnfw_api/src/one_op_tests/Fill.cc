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

TEST_F(GenModelTest, OneOp_Fill_Int32)
{
  CircleGen cgen;
  std::vector<int32_t> value_data{13};
  uint32_t value_buf = cgen.addBuffer(value_data);

  int in = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int value = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, value_buf});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorFill({{in, value}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<int32_t>({13, 13, 13, 13, 13, 13}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Fill_Int64)
{
  CircleGen cgen;
  std::vector<int64_t> value_data{13};
  uint32_t value_buf = cgen.addBuffer(value_data);

  int in = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int value = cgen.addTensor({{1}, circle::TensorType::TensorType_INT64, value_buf});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_INT64});
  cgen.addOperatorFill({{in, value}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<int64_t>({13, 13, 13, 13, 13, 13}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Fill_Float32)
{
  CircleGen cgen;
  std::vector<float> value_data{1.3};
  uint32_t value_buf = cgen.addBuffer(value_data);

  int in = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int value = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, value_buf});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFill({{in, value}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<float>({1.3, 1.3, 1.3, 1.3, 1.3, 1.3}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Fill_Int32_oneoperand)
{
  CircleGen cgen;

  int in = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorFill({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<int32_t>({13, 13, 13, 13, 13, 13}));
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Fill_Int64_oneoperand)
{
  CircleGen cgen;

  int in = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_INT64});
  cgen.addOperatorFill({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<int64_t>({13, 13, 13, 13, 13, 13}));
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Fill_Float32_oneoperand)
{
  CircleGen cgen;

  int in = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFill({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<int32_t>({2, 3}).addOutput<float>({1.3, 1.3, 1.3, 1.3, 1.3, 1.3}));
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
