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

TEST_F(GenModelTest, OneOp_AddN_1D)
{
  CircleGen cgen;

  int in1 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32});
  int in2 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32});
  int in3 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorAddN({{in1, in2, in3}, {out}});
  cgen.setInputsAndOutputs({in1, in2, in3}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->addTestCase(uniformTCD<float>({{1.2, 2.0, -3.0, 4.5, 10.0, 5.1, -7.0, 12.0},
                                           {3.3, 4.1, 3.0, 4.4, 5.0, 4.3, -1.2, 4.0},
                                           {-5.2, 3.1, 2.2, -3.7, 5.2, 2.0, -4.3, 5.0}},
                                          {{-0.7, 9.2, 2.2, 5.2, 20.2, 11.4, -12.5, 21.0}}));

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_AddN_InvalidType)
{
  CircleGen cgen;

  int in1 = cgen.addTensor({{8}, circle::TensorType::TensorType_UINT8});
  int in2 = cgen.addTensor({{8}, circle::TensorType::TensorType_UINT8});
  int in3 = cgen.addTensor({{8}, circle::TensorType::TensorType_UINT8});
  int out = cgen.addTensor({{8}, circle::TensorType::TensorType_UINT8});

  cgen.addOperatorAddN({{in1, in2, in3}, {out}});
  cgen.setInputsAndOutputs({in1, in2, in3}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_AddN_TypeDiff)
{
  CircleGen cgen;

  int in1 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32});
  int in2 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32});
  int in3 = cgen.addTensor({{8}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{8}, circle::TensorType::TensorType_INT32});

  cgen.addOperatorAddN({{in1, in2, in3}, {out}});
  cgen.setInputsAndOutputs({in1, in2, in3}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
