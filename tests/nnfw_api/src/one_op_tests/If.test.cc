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

TEST_F(GenModelTest, OneOp_If)
{
  // The model looks just like the below pseudocode
  //
  // function model(x)
  // {
  //   if (x < 0.0)
  //     return -100.0;
  //   else
  //     return 100.0;
  // }

  CircleGen cgen;

  // constant buffers
  std::vector<float> comp_data{0.0};
  uint32_t comp_buf = cgen.addBuffer(comp_data);
  std::vector<float> then_data{-100};
  uint32_t then_buf = cgen.addBuffer(then_data);
  std::vector<float> else_data{100};
  uint32_t else_buf = cgen.addBuffer(else_data);

  // primary subgraph
  {
    int x = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int comp = cgen.addTensor({{1}, circle::TensorType_FLOAT32, comp_buf});
    int cond = cgen.addTensor({{1}, circle::TensorType_BOOL});
    cgen.addOperatorLess({{x, comp}, {cond}});

    int ret = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorIf({{cond}, {ret}}, 1, 2);

    cgen.setInputsAndOutputs({x}, {ret});
  }

  // then subgraph
  {
    cgen.nextSubgraph();
    int ret = cgen.addTensor({{1}, circle::TensorType_FLOAT32, then_buf});
    cgen.setInputsAndOutputs({}, {ret});
  }

  // else subgraph
  {
    cgen.nextSubgraph();
    int ret = cgen.addTensor({{1}, circle::TensorType_FLOAT32, else_buf});
    cgen.setInputsAndOutputs({}, {ret});
  }

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{-1.0}}, {{-100.0}}));
  _context->addTestCase(uniformTCD<float>({{1.0}}, {{100.0}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

class IfWrongSubgraphIndex : public GenModelTest,
                             public ::testing::WithParamInterface<std::pair<int, int>>
{
};

TEST_P(IfWrongSubgraphIndex, neg_Test)
{
  // These values must be less than 0 or greater than 2
  int then_subg = GetParam().first;
  int else_subg = GetParam().second;

  // When If operation's subgraph index is invalid

  CircleGen cgen;

  // constant buffers
  std::vector<float> then_data{-100};
  uint32_t then_buf = cgen.addBuffer(then_data);
  std::vector<float> else_data{100};
  uint32_t else_buf = cgen.addBuffer(else_data);

  // primary subgraph
  {
    int x = cgen.addTensor({{1}, circle::TensorType_BOOL});
    int ret = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorIf({{x}, {ret}}, then_subg, else_subg);

    cgen.setInputsAndOutputs({x}, {ret});
  }

  // then subgraph
  {
    cgen.nextSubgraph();
    int ret = cgen.addTensor({{1}, circle::TensorType_FLOAT32, then_buf});
    cgen.setInputsAndOutputs({}, {ret});
  }

  // else subgraph
  {
    cgen.nextSubgraph();
    int ret = cgen.addTensor({{1}, circle::TensorType_FLOAT32, else_buf});
    cgen.setInputsAndOutputs({}, {ret});
  }

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(GenModelTest, IfWrongSubgraphIndex,
                         ::testing::Values(std::make_pair(99, 2), std::make_pair(-1, 2),
                                           std::make_pair(1, 99), std::make_pair(1, -99),
                                           std::make_pair(-99, 99)));
