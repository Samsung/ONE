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

CircleGen genSimpleSqrtModel(circle::TensorType type)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 2, 2, 1}, type});
  int out = cgen.addTensor({{1, 2, 2, 1}, type});
  cgen.addOperatorSqrt({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});
  return cgen;
}

TEST_F(GenModelTest, OneOp_Sqrt_f32)
{
  CircleGen cgen = genSimpleSqrtModel(circle::TensorType::TensorType_FLOAT32);

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput<float>({1, 4, 9, 16}).addOutput<float>({1, 2, 3, 4}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Sqrt_i32)
{
  CircleGen cgen = genSimpleSqrtModel(circle::TensorType::TensorType_INT32);

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}.addInput<int>({1, 4, 9, 16}).addOutput<float>({1, 2, 3, 4}));
  _context->setBackends({"cpu"});
  _context->expectFailCompile();

  SUCCEED();
}
