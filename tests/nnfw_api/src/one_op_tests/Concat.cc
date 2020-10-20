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

TEST_F(GenModelTest, OneOp_Concat_ShareSubTensor)
{
  CircleGen cgen;
  int lhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int rhs = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int shared_subtensor = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int concat_out = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> padding_data{0, 0, 1, 1, 1, 1, 0, 0};
  uint32_t padding_buf = cgen.addBuffer(padding_data);
  int padding = cgen.addTensor({{4, 2}, circle::TensorType::TensorType_INT32, padding_buf});
  int pad_out = cgen.addTensor({{1, 4, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAdd({{lhs, rhs}, {shared_subtensor}}, circle::ActivationFunctionType_NONE);
  cgen.addOperatorConcatenation({{rhs, shared_subtensor}, {concat_out}}, 3,
                                circle::ActivationFunctionType_NONE);
  cgen.addOperatorPad({{shared_subtensor, padding}, {pad_out}});
  cgen.setInputsAndOutputs({lhs, rhs}, {pad_out, concat_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>(
      {{1, 3, 2, 4}, {5, 4, 7, 4}},
      {{0, 0, 0, 0, 0, 6, 7, 0, 0, 9, 8, 0, 0, 0, 0, 0}, {5, 6, 4, 7, 7, 9, 4, 8}}));
  _context->setBackends({"acl_cl", "acl_neon"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Concat)
{
  CircleGen cgen;

  int input1 = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  int input2 = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  int output = cgen.addTensor({{4, 3}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorConcatenation({{input1, input2}, {output}}, 0,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({input1, input2}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4, 5, 6}, {7, 8, 9, 10, 11, 12}},
                                          {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}}));
  _context->setBackends({"cpu;acl_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Concat_Subtensor_4D)
{
  CircleGen cgen;
  int in1 = cgen.addTensor({{1, 1, 1, 20}, circle::TensorType::TensorType_FLOAT32});
  int in2 = cgen.addTensor({{1, 1, 1, 10}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> axis_data{3};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});

  int s_out1 = cgen.addTensor({{1, 1, 1, 5}, circle::TensorType::TensorType_FLOAT32});
  int s_out2 = cgen.addTensor({{1, 1, 1, 5}, circle::TensorType::TensorType_FLOAT32});
  int s_out3 = cgen.addTensor({{1, 1, 1, 5}, circle::TensorType::TensorType_FLOAT32});
  int s_out4 = cgen.addTensor({{1, 1, 1, 5}, circle::TensorType::TensorType_FLOAT32});

  int c_out1 = cgen.addTensor({{1, 1, 1, 10}, circle::TensorType::TensorType_FLOAT32});
  int c_out2 = cgen.addTensor({{1, 1, 1, 10}, circle::TensorType::TensorType_FLOAT32});
  int c_out3 = cgen.addTensor({{1, 1, 1, 10}, circle::TensorType::TensorType_FLOAT32});

  int a_out1 = cgen.addTensor({{1, 1, 1, 10}, circle::TensorType::TensorType_FLOAT32});
  int a_out2 = cgen.addTensor({{1, 1, 1, 10}, circle::TensorType::TensorType_FLOAT32});
  int a_out3 = cgen.addTensor({{1, 1, 1, 10}, circle::TensorType::TensorType_FLOAT32});

  int final_out = cgen.addTensor({{1, 1, 1, 35}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorSplit({{axis, in1}, {s_out1, s_out2, s_out3, s_out4}}, 4);

  cgen.addOperatorConcatenation({{s_out1, s_out2}, {c_out1}}, 3,
                                circle::ActivationFunctionType::ActivationFunctionType_NONE);
  cgen.addOperatorConcatenation({{s_out1, s_out3}, {c_out2}}, 3,
                                circle::ActivationFunctionType::ActivationFunctionType_NONE);
  cgen.addOperatorConcatenation({{s_out1, s_out4}, {c_out3}}, 3,
                                circle::ActivationFunctionType::ActivationFunctionType_NONE);

  cgen.addOperatorAdd({{c_out1, in2}, {a_out1}},
                      circle::ActivationFunctionType::ActivationFunctionType_NONE);
  cgen.addOperatorAdd({{c_out2, in2}, {a_out2}},
                      circle::ActivationFunctionType::ActivationFunctionType_NONE);
  cgen.addOperatorAdd({{c_out3, in2}, {a_out3}},
                      circle::ActivationFunctionType::ActivationFunctionType_NONE);

  cgen.addOperatorConcatenation({{s_out1, a_out1, a_out2, a_out3}, {final_out}}, 3,
                                circle::ActivationFunctionType::ActivationFunctionType_NONE);

  cgen.setInputsAndOutputs({in1, in2}, {s_out1, s_out2, s_out3, s_out4, c_out1, c_out2, c_out3,
                                        a_out1, a_out2, a_out3, final_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>(
      {
          // inputs
          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}, // in1
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}                                           // in2
      },
      {
          // outputs
          {1, 2, 3, 4, 5},                     // s_out1
          {6, 7, 8, 9, 10},                    // s_out2
          {11, 12, 13, 14, 15},                // s_out3
          {16, 17, 18, 19, 20},                // s_out4
          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},     // c_out1
          {1, 2, 3, 4, 5, 11, 12, 13, 14, 15}, // c_out2
          {1, 2, 3, 4, 5, 16, 17, 18, 19, 20}, // c_out3
          {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},     // a_out1
          {1, 2, 3, 4, 5, 11, 12, 13, 14, 15}, // a_out2
          {1, 2, 3, 4, 5, 16, 17, 18, 19, 20}, // a_out3
          {1, 2, 3,  4,  5,  1,  2,  3, 4, 5, 6, 7, 8,  9,  10, 1,  2, 3,
           4, 5, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20} // final_out
      }));
  _context->setBackends({"acl_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Concat_Subtensor_2D)
{
  CircleGen cgen;
  int in1 = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_FLOAT32});
  int in2 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
  std::vector<int32_t> axis_data{1};
  uint32_t axis_buf = cgen.addBuffer(axis_data);
  int axis = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, axis_buf});

  int s_out1 = cgen.addTensor({{1, 1}, circle::TensorType::TensorType_FLOAT32});
  int s_out2 = cgen.addTensor({{1, 1}, circle::TensorType::TensorType_FLOAT32});
  int s_out3 = cgen.addTensor({{1, 1}, circle::TensorType::TensorType_FLOAT32});
  int s_out4 = cgen.addTensor({{1, 1}, circle::TensorType::TensorType_FLOAT32});

  int c_out1 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
  int c_out2 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
  int c_out3 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});

  int a_out1 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
  int a_out2 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});
  int a_out3 = cgen.addTensor({{1, 2}, circle::TensorType::TensorType_FLOAT32});

  int final_out = cgen.addTensor({{1, 7}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorSplit({{axis, in1}, {s_out1, s_out2, s_out3, s_out4}}, 4);

  cgen.addOperatorConcatenation({{s_out1, s_out2}, {c_out1}}, 1,
                                circle::ActivationFunctionType::ActivationFunctionType_NONE);
  cgen.addOperatorConcatenation({{s_out1, s_out3}, {c_out2}}, 1,
                                circle::ActivationFunctionType::ActivationFunctionType_NONE);
  cgen.addOperatorConcatenation({{s_out1, s_out4}, {c_out3}}, 1,
                                circle::ActivationFunctionType::ActivationFunctionType_NONE);

  cgen.addOperatorAdd({{c_out1, in2}, {a_out1}},
                      circle::ActivationFunctionType::ActivationFunctionType_NONE);
  cgen.addOperatorAdd({{c_out2, in2}, {a_out2}},
                      circle::ActivationFunctionType::ActivationFunctionType_NONE);
  cgen.addOperatorAdd({{c_out3, in2}, {a_out3}},
                      circle::ActivationFunctionType::ActivationFunctionType_NONE);

  cgen.addOperatorConcatenation({{s_out1, a_out1, a_out2, a_out3}, {final_out}}, 1,
                                circle::ActivationFunctionType::ActivationFunctionType_NONE);

  cgen.setInputsAndOutputs({in1, in2}, {s_out1, s_out2, s_out3, s_out4, c_out1, c_out2, c_out3,
                                        a_out1, a_out2, a_out3, final_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>(
      {
          // inputs
          {1, 2, 3, 4}, // in1
          {0, 0}        // in2
      },
      {
          // outputs
          {1},                  // s_out1
          {2},                  // s_out2
          {3},                  // s_out3
          {4},                  // s_out4
          {1, 2},               // c_out1
          {1, 3},               // c_out2
          {1, 4},               // c_out3
          {1, 2},               // a_out1
          {1, 3},               // a_out2
          {1, 4},               // a_out3
          {1, 1, 2, 1, 3, 1, 4} // final_out
      }));
  _context->setBackends({"acl_cl"});

  SUCCEED();
}
