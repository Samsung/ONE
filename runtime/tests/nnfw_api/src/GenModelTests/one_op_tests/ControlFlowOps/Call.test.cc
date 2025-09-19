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

TEST_F(GenModelTest, OneOp_Call)
{
  // The model looks just like the below pseudocode
  //
  // function model(x)
  // {
  //   return x+1;
  // }

  CircleGen cgen;
  uint32_t incr_buf = cgen.addBuffer(std::vector<float>{1});

  // primary subgraph
  {
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorCall({{x_in}, {x_out}}, 1);
    cgen.setInputsAndOutputs({x_in}, {x_out});
  }

  // callee subgraph
  {
    cgen.nextSubgraph();
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int incr = cgen.addTensor({{1}, circle::TensorType_FLOAT32, incr_buf});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorAdd({{x_in, incr}, {x_out}}, circle::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({x_in}, {x_out});
  }

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0}}, {{1}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}
