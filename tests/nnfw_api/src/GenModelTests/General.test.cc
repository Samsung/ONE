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

/**
 * @file This file contains miscellaneous GenModelTest test cases.
 *
 */

#include "GenModelTest.h"

#include <memory>

TEST_F(GenModelTest, UnusedConstOutputOnly)
{
  // A single tensor which is constant
  CircleGen cgen;
  uint32_t const_buf = cgen.addBuffer(std::vector<float>{9, 8, 7, 6});
  int out_const = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, const_buf});
  cgen.setInputsAndOutputs({}, {out_const});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({}, {{9, 8, 7, 6}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, UnusedConstOutputAndAdd)
{
  // A single tensor which is constant + an Add op
  CircleGen cgen;
  uint32_t rhs_buf = cgen.addBuffer(std::vector<float>{5, 4, 7, 4});
  uint32_t const_buf = cgen.addBuffer(std::vector<float>{9, 8, 7, 6});
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, rhs_buf});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out_const = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, const_buf});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs}, {out, out_const});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 3, 2, 4}}, {{6, 7, 9, 8}, {9, 8, 7, 6}}));
  _context->addTestCase(uniformTCD<float>({{0, 1, 2, 3}}, {{5, 5, 9, 7}, {9, 8, 7, 6}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, UsedConstOutput)
{
  // (( Input 1 )) ---------\
  //                         |=> [ Add ] -> (( Output 1 ))
  // (( Const Output 2 )) --<
  //                         |=> [ Add ] -> (( Output 0 ))
  // (( Input 0 )) ---------/
  CircleGen cgen;
  uint32_t rhs_buf = cgen.addBuffer(std::vector<float>{6, 4, 8, 1});
  int in0 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int in1 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out0 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out1 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int const_out2 = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, rhs_buf});
  cgen.addOperatorAdd({{in0, const_out2}, {out0}}, circle::ActivationFunctionType_NONE);
  cgen.addOperatorAdd({{const_out2, in1}, {out1}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in0, in1}, {out0, out1, const_out2});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 1, 1, 1}, {-1, -1, -1, -1}},
                                          {{7, 5, 9, 2}, {5, 3, 7, 0}, {6, 4, 8, 1}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, TensorBothInputOutput)
{
  // A single tensor which is an input and an output at the same time
  CircleGen cgen;
  int t = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.setInputsAndOutputs({t}, {t});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 3, 2, 4}}, {{1, 3, 2, 4}}));
  _context->addTestCase(uniformTCD<float>({{100, 300, 200, 400}}, {{100, 300, 200, 400}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, TensorBothInputOutputCrossed)
{
  // Two tensors which are an input and an output at the same time
  // But the order of inputs and outputs is changed.
  CircleGen cgen;
  int t1 = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int t2 = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  cgen.setInputsAndOutputs({t1, t2}, {t2, t1});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1}, {2}}, {{2}, {1}}));
  _context->addTestCase(uniformTCD<float>({{100}, {200}}, {{200}, {100}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneTensor_TwoOutputs)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out, out}); // Same tensors are used twice as output

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 1}, {2, 2}}, {{3, 3}, {3, 3}}));
  _context->addTestCase(uniformTCD<float>({{2, 4}, {7, 4}}, {{9, 8}, {9, 8}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneTensor_ThreeOutputs)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {out}}, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({lhs, rhs}, {out, out, out}); // Same tensors are used 3 times as output

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1}, {2}}, {{3}, {3}, {3}}));
  _context->addTestCase(uniformTCD<float>({{2}, {7}}, {{9}, {9}, {9}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneTensor_InputAndTwoOutputs)
{
  CircleGen cgen;
  int t = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  cgen.setInputsAndOutputs({t}, {t, t}); // Same tensor is an input and 2 outputs

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 1}}, {{1, 1}, {1, 1}}));
  _context->addTestCase(uniformTCD<float>({{2, 4}}, {{2, 4}, {2, 4}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneTensor_InputAndTwoOutputsUsed)
{
  CircleGen cgen;
  int t = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  int o = cgen.addTensor({{2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorNeg({{t}, {o}});
  cgen.setInputsAndOutputs({t}, {t, t, o}); // Same tensor is an input and 2 outputs

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 1}}, {{1, 1}, {1, 1}, {-1, -1}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneTensor_ConstAndThreeOutputs)
{
  CircleGen cgen;
  uint32_t const_buf = cgen.addBuffer(std::vector<float>{2, 5});
  int t = cgen.addTensor({{2}, circle::TensorType_FLOAT32, const_buf});
  cgen.setInputsAndOutputs({}, {t, t, t}); // A const tensor is 3 outputs

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({}, {{2, 5}, {2, 5}, {2, 5}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, Reshape_with_shape_param_as_const)
{
  CircleGen cgen;
  auto i32 = circle::TensorType::TensorType_INT32;

  int input = cgen.addTensor({{4}, i32});

  std::vector<int32_t> new_shape_data{2, 2}; // const of value [2, 2]
  uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  int new_shape = cgen.addTensor({{2}, i32, new_shape_buf});

  int out = cgen.addTensor({{2, 2}, i32});

  // reshape with new_shape param
  cgen.addOperatorReshape({{input, new_shape}, {out}}, &new_shape_data);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<int32_t>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_Reshape_with_shape_param_as_const)
{
  // We will ses if Reshape with shape param can generate error during compilation if param is wrong
  CircleGen cgen;
  auto i32 = circle::TensorType::TensorType_INT32;

  int input = cgen.addTensor({{4}, i32});

  std::vector<int32_t> wrong_new_shape_data{2, 3}; // not match with input shape
  uint32_t new_shape_buf = cgen.addBuffer(wrong_new_shape_data);
  int new_shape = cgen.addTensor({{2}, i32, new_shape_buf});

  int out = cgen.addTensor({{2, 2}, i32});

  cgen.addOperatorReshape({{input, new_shape}, {out}}, &wrong_new_shape_data);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<int32_t>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, Reshape_with_shape_param_as_const_float)
{
  CircleGen cgen;
  auto f32 = circle::TensorType::TensorType_FLOAT32;
  int input = cgen.addTensor({{4}, f32});

  std::vector<int32_t> new_shape_data{2, 2}; // const of value [2, 2]
  uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  int new_shape = cgen.addTensor({{2}, f32, new_shape_buf});
  int out = cgen.addTensor({{2, 2}, f32});

  // reshape with new_shape param
  cgen.addOperatorReshape({{input, new_shape}, {out}}, &new_shape_data);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"gpu_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_Reshape_with_shape_param_as_const_float)
{
  // We will ses if Reshape with shape param can generate error during compilation if param is wrong
  CircleGen cgen;
  auto f32 = circle::TensorType::TensorType_FLOAT32;

  int input = cgen.addTensor({{4}, f32});

  std::vector<int32_t> wrong_new_shape_data{2, 3}; // not match with input shape
  uint32_t new_shape_buf = cgen.addBuffer(wrong_new_shape_data);
  int new_shape = cgen.addTensor({{2}, f32, new_shape_buf});

  int out = cgen.addTensor({{2, 2}, f32});

  cgen.addOperatorReshape({{input, new_shape}, {out}}, &wrong_new_shape_data);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"gpu_cl"});

  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, Reshape_without_shape_param)
{
  CircleGen cgen;
  auto i32 = circle::TensorType::TensorType_INT32;

  int input = cgen.addTensor({{4}, i32});
  int new_shape = cgen.addTensor({{2}, i32}); // reshape to 2D tensor
  int out = cgen.addTensor({{}, i32}); // exact shape is not unknown since ouput is dynamic tensor

  // reshape with new_shape param
  cgen.addOperatorReshape({{input, new_shape}, {out}} /* no new_shape param */);
  cgen.setInputsAndOutputs({input, new_shape}, {out});

  CircleGen::Shape new_shape_val{2, 2};
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<int32_t>({{1, 2, 3, 4}, new_shape_val}, {{1, 2, 3, 4}}));
  _context->output_sizes(0, sizeof(int32_t) * 4);
  _context->setBackends({"cpu" /* "acl_cl", "acl_neon" does not support dynamic tensor */});

  SUCCEED();
}

TEST_F(GenModelTest, neg_Reshape_without_shape_param)
{
  // We will ses if Reshape without shape param can generate error whiile running
  CircleGen cgen;
  auto i32 = circle::TensorType::TensorType_INT32;

  int input = cgen.addTensor({{4}, i32});
  int new_shape = cgen.addTensor({{2}, i32}); // reshape to 2D tensor
  int out = cgen.addTensor({{}, i32}); // exact shape is not unknown since ouput is dynamic tensor

  // reshape with new_shape param
  cgen.addOperatorReshape({{input, new_shape}, {out}} /* no new_shape param */);
  cgen.setInputsAndOutputs({input, new_shape}, {out});

  CircleGen::Shape wrong_new_shape_val{2, 3};
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  auto tc = uniformTCD<int32_t>({{1, 2, 3, 4}, wrong_new_shape_val}, {{1, 2, 3, 4}});
  tc.expectFailRun();
  _context->addTestCase(tc);
  _context->setBackends({"cpu" /* "acl_cl", "acl_neon" does not support dynamic tensor */});

  SUCCEED();
}

// test to check model that has op->while->op
TEST_F(GenModelTest, while_with_input_output)
{
  // The model looks just like the below pseudocode
  //
  //   x = cast(int to float)
  //   while (x < 100.0)
  //   {
  //     x = x + 10.0;
  //   }
  //   x = cast(float to int)

  CircleGen cgen;
  std::vector<float> incr_data{10};
  uint32_t incr_buf = cgen.addBuffer(incr_data);
  std::vector<float> end_data{100};
  uint32_t end_buf = cgen.addBuffer(end_data);

  // primary subgraph
  {
    int model_in = cgen.addTensor({{1}, circle::TensorType_INT32});
    int cast_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int while_out = cgen.addTensor({{1}, circle::TensorType_FLOAT32});
    int model_out = cgen.addTensor({{1}, circle::TensorType_INT32});

    cgen.addOperatorCast({{model_in}, {cast_out}}, circle::TensorType_INT32,
                         circle::TensorType_FLOAT32);
    cgen.addOperatorWhile({{cast_out}, {while_out}}, 1, 2);
    cgen.addOperatorCast({{while_out}, {model_out}}, circle::TensorType_FLOAT32,
                         circle::TensorType_INT32);

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
    cgen.addOperatorAdd({{x_in, incr}, {x_out}}, circle::ActivationFunctionType_NONE);
    cgen.setInputsAndOutputs({x_in}, {x_out});
  }

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<int>({{0}}, {{100}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}
