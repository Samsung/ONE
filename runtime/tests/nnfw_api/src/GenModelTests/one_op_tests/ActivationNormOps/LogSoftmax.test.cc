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

TEST_F(GenModelTest, OneOp_LogSoftmax)
{
  // NOTE For tf lite the params are fixed as:
  // beta = 1.0, axis = -1

  CircleGen cgen;
  int in = cgen.addTensor({{1, 1, 1, 4, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1, 4, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorLogSoftmax({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->addTestCase(uniformTCD<float>(
    {{0, -6, 2, 4, 3, -2, 10, 1}},
    {{-.00247565, -6.00247, -2.12692, -.126928, -.00671534, -5.00671, -.000123374, -9.00012}}));

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_LogSoftmax_InvalidModel)
{
  CircleGen cgen;
  int out = cgen.addTensor({{4, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorLogSoftmax({{}, {out}}); // No input tensor
  cgen.setInputsAndOutputs({}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
