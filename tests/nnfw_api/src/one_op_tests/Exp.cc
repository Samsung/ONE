/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

TEST_F(GenModelTest, OneOp_Exp_Float32)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorExp({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{3.0, 4.0, 5.0, 6.0}}, {{20.085537, 54.59815, 148.41316, 403.4288}}));
  _context->setBackends({"cpu", "gpu_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Exp_Float32_TwoOperand)
{
  CircleGen cgen;
  int in1 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int in2 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out1 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out2 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorExp({{in1, in2}, {out1, out2}});
  cgen.setInputsAndOutputs({in1, in2}, {out1, out2});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu", "gpu_cl"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Exp_Int32_TwoOperand)
{
  CircleGen cgen;
  int in1 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  int in2 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  int out1 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  int out2 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorExp({{in1, in2}, {out1, out2}});
  cgen.setInputsAndOutputs({in1, in2}, {out1, out2});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu", "gpu_cl"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Exp_Int64_TwoOperand)
{
  CircleGen cgen;
  int in1 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT64});
  int in2 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT64});
  int out1 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT64});
  int out2 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT64});
  cgen.addOperatorExp({{in1, in2}, {out1, out2}});
  cgen.setInputsAndOutputs({in1, in2}, {out1, out2});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu", "gpu_cl"});
  _context->expectFailModelLoad();

  SUCCEED();
}
