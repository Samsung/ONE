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

TEST_F(GenModelTest, OneOp_Relu6)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorRelu6({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{4, 7.0, 3.0, 8.0, -1.0, -2.0f}}, {{4, 6.0, 3.0, 6.0, 0, 0}}));
  _context->setBackends({"cpu", "gpu_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Relu6_InvalidType)
{
  CircleGen cgen;
  int in = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_UINT8});
  int out = cgen.addTensor({{2, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorRelu6({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu", "gpu_cl"});
  _context->expectFailModelLoad();

  SUCCEED();
}
