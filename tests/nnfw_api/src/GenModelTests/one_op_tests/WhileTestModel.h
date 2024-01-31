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

#ifndef __NNFW_API_TEST_WHILE_TEST_MODEL_H__
#define __NNFW_API_TEST_WHILE_TEST_MODEL_H__

#include "GenModelTest.h"

#include <memory>

class WhileModelLoop10
{
public:
  WhileModelLoop10()
  {
    // The model looks just like the below pseudocode
    //
    // function model(x)
    // {
    //   while (x < 100.0)
    //   {
    //     x = x + 10.0;
    //   }
    //   return x
    // }
    CircleGen cgen;
    std::vector<float> incr_data{10};
    uint32_t incr_buf = cgen.addBuffer(incr_data);
    std::vector<float> end_data{100};
    uint32_t end_buf = cgen.addBuffer(end_data);

    // primary subgraph
    {
      int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
      int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
      cgen.addOperatorWhile({{x_in}, {x_out}}, 1, 2);
      cgen.setInputsAndOutputs({x_in}, {x_out});
    }

    // cond subgraph
    {
      cgen.nextSubgraph();
      int x = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
      int end = cgen.addTensor({{1}, circle::TensorType_FLOAT32, end_buf});
      int result = cgen.addTensor({{1}, circle::TensorType_BOOL});
      cgen.addOperatorLess({{x, end}, {result}});
      cgen.setInputsAndOutputs({x}, {result});
    }

    // body subgraph
    {
      cgen.nextSubgraph();
      int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
      int incr = cgen.addTensor({{1}, circle::TensorType_FLOAT32, incr_buf});
      int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
      cgen.addOperatorAdd({{x_in, incr}, {x_out}}, circle::ActivationFunctionType_NONE);
      cgen.setInputsAndOutputs({x_in}, {x_out});
    }
    cbuf = cgen.finish();
  }

  int inputCount() { return 1; }
  int outputputCount() { return 1; }
  int sizeOfDType() { return sizeof(float); }

  CircleBuffer cbuf;
};

#endif // __NNFW_API_TEST_WHILE_TEST_MODEL_H__
