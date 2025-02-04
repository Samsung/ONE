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

TEST_F(GenModelTest, OneOp_Elu)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorElu({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{0, -6, 2, -4, 3, -2, 10, -0.1}},
                      {{0.0, -0.997521, 2.0, -0.981684, 3.0, -0.864665, 10.0, -0.0951626}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Elu_Type)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_UINT8}, 1.0f, 0);
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorElu({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
