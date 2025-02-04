/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

TEST_F(GenModelTest, OneOp_BatchToSpaceND_notCrop_1x1)
{
  CircleGen cgen;
  int in = cgen.addTensor({{4, 1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int block = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorBatchToSpaceND({{in, block}, {out}});
  cgen.setInputsAndOutputs({in, block}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<float>({1, 2, 3, 4})
                          .addInput<int32_t>({2, 2})
                          .addOutput<float>({1, 2, 3, 4}));
  SUCCEED();
}

TEST_F(GenModelTest, OneOp_BatchToSpaceND_notCrop_2x2)
{
  CircleGen cgen;
  int in = cgen.addTensor({{4, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 4, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  int block = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorBatchToSpaceND({{in, block}, {out}});
  cgen.setInputsAndOutputs({in, block}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}
      .addInput<float>({1, 3, 9, 11, 2, 4, 10, 12, 5, 7, 13, 15, 6, 8, 14, 16})
      .addInput<int32_t>({2, 2})
      .addOutput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}));
  _context->setBackends({"cpu"});
  SUCCEED();
}

TEST_F(GenModelTest, OneOp_BatchToSpaceND_Crop)
{
  CircleGen cgen;
  int in = cgen.addTensor({{8, 1, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2, 2, 4, 1}, circle::TensorType::TensorType_FLOAT32});
  int block = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  int crop = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorBatchToSpaceND({{in, block, crop}, {out}});
  cgen.setInputsAndOutputs({in, block, crop}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}
      .addInput<float>(
        {0, 1, 3, 0, 9, 11, 0, 2, 4, 0, 10, 12, 0, 5, 7, 0, 13, 15, 0, 6, 8, 0, 14, 16})
      .addInput<int32_t>({2, 2})
      .addInput<int32_t>({0, 0, 2, 0})
      .addOutput<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}));
  _context->setBackends({"cpu"});
  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_BatchToSpaceND_DifferentType)
{
  CircleGen cgen;
  int in = cgen.addTensor({{4, 1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_INT32});
  int block = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32});
  cgen.addOperatorBatchToSpaceND({{in, block}, {out}});
  cgen.setInputsAndOutputs({in, block}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(TestCaseData{}
                          .addInput<float>({1, 2, 3, 4})
                          .addInput<int32_t>({2, 2})
                          .addOutput<int>({1, 2, 3, 4}));
  _context->expectFailModelLoad();
  SUCCEED();
}
