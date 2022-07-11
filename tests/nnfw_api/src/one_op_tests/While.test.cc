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
#include "WhileTestModel.h"

#include <memory>

TEST_F(GenModelTest, OneOp_While)
{
  WhileModelLoop10 model;
  _context = std::make_unique<GenModelTestContext>(std::move(model.cbuf));
  _context->addTestCase(uniformTCD<float>({{0}}, {{100}}));
  _context->addTestCase(uniformTCD<float>({{2}}, {{102}}));
  _context->addTestCase(uniformTCD<float>({{22}}, {{102}}));
  _context->addTestCase(uniformTCD<float>({{100}}, {{100}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_While_github_4783)
{
  // The model looks just like the below pseudocode
  //
  // function model(x, data)
  // {
  //   // `data` does not do anything but passed to while's cond and body subgraphs
  //   // to measure copy overhead between subgraphs
  //   while (x < 100.0)
  //   {
  //     x = x + 1.0;
  //   }
  //   return (x, data)
  // }

  const int kElems = 4;
  const std::vector<int32_t> shape{kElems};

  CircleGen cgen;
  uint32_t incr_buf = cgen.addBuffer(std::vector<float>{1});
  uint32_t incr_data_buf = cgen.addBuffer(std::vector<float>(kElems, 1));
  uint32_t end_buf = cgen.addBuffer(std::vector<float>{100});

  // primary subgraph
  {
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int d_in = cgen.addTensor({shape, circle::TensorType_FLOAT32});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int d_out = cgen.addTensor({shape, circle::TensorType_FLOAT32});
    cgen.addOperatorWhile({{x_in, d_in}, {x_out, d_out}}, 1, 2);
    cgen.setInputsAndOutputs({x_in, d_in}, {x_out, d_out});
  }

  // cond subgraph
  {
    cgen.nextSubgraph();
    int x = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int d = cgen.addTensor({shape, circle::TensorType_FLOAT32});
    int end = cgen.addTensor({{1}, circle::TensorType_FLOAT32, end_buf});
    int result = cgen.addTensor({{1}, circle::TensorType_BOOL});
    cgen.addOperatorLess({{x, end}, {result}});
    cgen.setInputsAndOutputs({x, d}, {result});
  }

  // body subgraph
  {
    cgen.nextSubgraph();
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int incr = cgen.addTensor({{1}, circle::TensorType_FLOAT32, incr_buf});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int d_in = cgen.addTensor({shape, circle::TensorType_FLOAT32});
    int incr_d = cgen.addTensor({shape, circle::TensorType_FLOAT32, incr_data_buf});
    int d_out = cgen.addTensor({shape, circle::TensorType_FLOAT32});
    cgen.addOperatorAdd({{x_in, incr}, {x_out}}, circle::ActivationFunctionType_NONE);
    cgen.addOperatorAdd({{d_in, incr_d}, {d_out}}, circle::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({x_in, d_in}, {x_out, d_out});
  }

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  std::vector<float> tc_data_in(kElems, 9);
  std::vector<float> tc_data_out(kElems, 109);
  _context->addTestCase(uniformTCD<float>({{0}, tc_data_in}, {{100}, tc_data_out}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_While_TwoInputs)
{
  // The model looks just like the below pseudocode
  //
  // function model(x, end)
  // {
  //   while (x < end)
  //   {
  //     x = x + 10.0
  //   }
  //   return x
  // }

  CircleGen cgen;
  std::vector<float> incr_data{10};
  uint32_t incr_buf = cgen.addBuffer(incr_data);

  // primary subgraph
  {
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorWhile({{x_in, end_in}, {x_out, end_out}}, 1, 2);
    cgen.setInputsAndOutputs({x_in, end_in}, {x_out});
  }

  // cond subgraph
  {
    cgen.nextSubgraph();
    int x = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int result = cgen.addTensor({{1}, circle::TensorType_BOOL});
    cgen.addOperatorLess({{x, end}, {result}});
    cgen.setInputsAndOutputs({x, end}, {result});
  }

  // body subgraph
  {
    cgen.nextSubgraph();
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int incr = cgen.addTensor({{1}, circle::TensorType_FLOAT32, incr_buf});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorAdd({{x_in, incr}, {x_out}}, circle::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({x_in, end}, {x_out, end});
  }

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0}, {20}}, {{20}}));
  _context->addTestCase(uniformTCD<float>({{5}, {30}}, {{35}}));
  _context->addTestCase(uniformTCD<float>({{20}, {10}}, {{20}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

class WhileWrongSubgraphIndex : public GenModelTest,
                                public ::testing::WithParamInterface<std::pair<int, int>>
{
};

TEST_P(WhileWrongSubgraphIndex, neg_Test)
{
  // These values must be less than 0 or greater than 2
  int cond_subg = GetParam().first;
  int body_subg = GetParam().second;

  // When While operation's subgraph index is invalid

  CircleGen cgen;

  // constant buffers
  std::vector<float> incr_data{10};
  uint32_t incr_buf = cgen.addBuffer(incr_data);

  // primary subgraph
  {
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorWhile({{x_in, end_in}, {x_out, end_out}}, cond_subg, body_subg);
    cgen.setInputsAndOutputs({x_in, end_in}, {x_out});
  }

  // cond subgraph
  {
    cgen.nextSubgraph();
    int x = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int result = cgen.addTensor({{1}, circle::TensorType_BOOL});
    cgen.addOperatorLess({{x, end}, {result}});
    cgen.setInputsAndOutputs({x, end}, {result});
  }

  // body subgraph
  {
    cgen.nextSubgraph();
    int x_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int incr = cgen.addTensor({{1}, circle::TensorType_FLOAT32, incr_buf});
    int x_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int end = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    cgen.addOperatorAdd({{x_in, incr}, {x_out}}, circle::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({x_in, end}, {x_out, end});
  }

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

INSTANTIATE_TEST_SUITE_P(GenModelTest, WhileWrongSubgraphIndex,
                         ::testing::Values(std::make_pair(99, 2), std::make_pair(-1, 2),
                                           std::make_pair(1, 99), std::make_pair(1, -99),
                                           std::make_pair(-99, 99)));

// In this test, output of WHILE and body subgraph have different data types
TEST_F(GenModelTest, neg_while_wrong_dtype)
{
  CircleGen cgen;
  std::vector<float> incr_data{10};
  uint32_t incr_buf = cgen.addBuffer(incr_data);
  std::vector<float> end_data{100};
  uint32_t end_buf = cgen.addBuffer(end_data);

  // primary subgraph
  {
    int model_in = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int model_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});

    cgen.addOperatorWhile({{model_in}, {model_out}}, 1, 2);
    cgen.setInputsAndOutputs({model_in}, {model_out});
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
    int cast_out = cgen.addTensor({{1}, circle::TensorType_INT32});
    cgen.addOperatorAdd({{x_in, incr}, {x_out}}, circle::ActivationFunctionType_NONE);
    cgen.addOperatorCast({{x_out}, {cast_out}}, circle::TensorType_FLOAT32,
                         circle::TensorType_INT32);
    cgen.setInputsAndOutputs({x_in}, {cast_out});
    // output of this subgraph is INT32 but output of WHILE is FLOAT32
  }

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  // It is correct to call `_context->expectFailModelLoad();`, but OperationValidator does not deal
  // with subgraphs. So it is verified by `_context->expectFailCompile(); as a workaround`
  _context->expectFailCompile();

  SUCCEED();
}
