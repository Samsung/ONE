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

CircleGen genSimpleQuantizeModel(circle::TensorType from_t, float input_scale, int input_zeropoint,
                                 circle::TensorType to_t, float output_scale, int output_zeropoint)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 4, 4, 1}, from_t}, input_scale, input_zeropoint);
  int out = cgen.addTensor({{1, 4, 4, 1}, to_t}, output_scale, output_zeropoint);
  cgen.addOperatorQuantize({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});
  return cgen;
}

TEST_F(GenModelTest, OneOp_Quantize_Uint8toInt8)
{
  CircleGen cgen =
    genSimpleQuantizeModel(circle::TensorType_UINT8, 1., 128, circle::TensorType_INT8, 2., -10);
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}
      .addInput<uint8_t>({127, 48, 151, 232, 56, 176, 47, 37, 51, 52, 39, 94, 15, 108, 142, 243})
      .addOutput<int8_t>(
        {-10, -50, 2, 42, -46, 14, -50, -55, -48, -48, -54, -27, -66, -20, -3, 48}));
  _context->setBackends({"cpu"});
  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Quantize_Int8toUint8)
{
  CircleGen cgen =
    genSimpleQuantizeModel(circle::TensorType_INT8, 2., -10, circle::TensorType_UINT8, 1., 128);
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}
      .addInput<int8_t>({-10, -50, 2, 42, -46, 14, -50, -55, -48, -48, -54, -27, -66, -20, -3, 48})
      .addOutput<uint8_t>({128, 48, 152, 232, 56, 176, 48, 38, 52, 52, 40, 94, 16, 108, 142, 244}));
  _context->setBackends({"cpu"});
  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Quantize_Uint8toInt16)
{
  CircleGen cgen =
    genSimpleQuantizeModel(circle::TensorType_UINT8, 1., 128, circle::TensorType_INT16, 2., -10);
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Quantize_Int8toInt16)
{
  CircleGen cgen =
    genSimpleQuantizeModel(circle::TensorType_INT8, 2., -10, circle::TensorType_INT16, 1., 128);
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
