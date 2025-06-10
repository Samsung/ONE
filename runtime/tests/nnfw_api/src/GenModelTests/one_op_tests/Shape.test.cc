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

// WORKAROUND Handle int32_t type input/output
TEST_F(GenModelTest, OneOp_Shape)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{4}, circle::TensorType::TensorType_INT32});

  cgen.addOperatorShape({{in}, {out}}, circle::TensorType::TensorType_INT32);
  cgen.setInputsAndOutputs({in}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}
      .addInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})
      .addOutput<int32_t>({1, 3, 3, 2}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Shape_Int64)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_INT64});

  // TODO handle many type in addTestCase
  cgen.addOperatorShape({{in}, {out}}, circle::TensorType::TensorType_INT64);
  cgen.setInputsAndOutputs({in}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}
      .addInput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18})
      .addOutput<int64_t>({1, 3, 3, 2}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Shape_OutType)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_UINT8});

  // TODO handle many type in addTestCase
  cgen.addOperatorShape({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
