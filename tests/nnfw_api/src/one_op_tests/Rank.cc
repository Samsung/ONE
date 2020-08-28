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
union float_int {
  int32_t i;
  float f;
};

TEST_F(GenModelTest, OneOp_Rank)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32});

  // TODO handle many type in addTestCase
  float_int output_data;
  output_data.i = 4;

  cgen.addOperatorRank({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
      {{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}}, {{output_data.f}}});
  _context->setBackends({"cpu"});
}

TEST_F(GenModelTest, OneOp_Rank_Int32)
{
  CircleGen cgen;
  int in = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_INT32});
  int out = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32});

  // TODO handle many type in addTestCase
  float_int output_data;
  output_data.i = 4;

  cgen.addOperatorRank({{in}, {out}});
  cgen.setInputsAndOutputs({in}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
      {{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}}, {{output_data.f}}});
  _context->setBackends({"cpu"});
}
