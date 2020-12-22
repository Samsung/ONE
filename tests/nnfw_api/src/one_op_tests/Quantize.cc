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

CircleGen genSimpleQuantizeModel(circle::TensorType from_t, circle::TensorType to_t)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 4, 4, 1}, from_t}, 1, 128);
  int out = cgen.addTensor({{1, 4, 4, 1}, to_t}, 2, -10);
  cgen.addOperatorQuantize({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});
  return cgen;
}

TEST_F(GenModelTest, OneOp_Quantize_Uint8toInt8)
{
  CircleGen cgen = genSimpleQuantizeModel(circle::TensorType_UINT8, circle::TensorType_INT8);
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}
      .addInput<uint8_t>({127, 48, 151, 232, 56, 176, 47, 37, 51, 52, 39, 94, 15, 108, 142, 243})
      .addOutput<int8_t>(
        {-10, -50, 2, 42, -46, 14, -50, -55, -48, -48, -54, -27, -66, -20, -3, 48}));
  _context->setBackends({"cpu"});
  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Quantize_Uint8toInt16)
{
  CircleGen cgen = genSimpleQuantizeModel(circle::TensorType_UINT8, circle::TensorType_INT16);
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}
      .addInput<uint8_t>({127, 48, 151, 232, 56, 176, 47, 37, 51, 52, 39, 94, 1, 128, 142, 243})
      .addOutput<int16_t>(
        {-1, -80, 23, 104, -72, 48, -81, -91, -77, -76, -89, -34, -127, 0, 14, 115}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
