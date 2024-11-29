/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

TEST_F(GenModelTest, OneOp_RoPE)
{
  CircleGen cgen;
  uint32_t sin_table_buf = cgen.addBuffer(std::vector<float>{0.5, 1.0, 1.0, 0.5});
  int sin_table =
    cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, sin_table_buf});
  uint32_t cos_table_buf = cgen.addBuffer(std::vector<float>{1.0, 0.5, 0.5, 1.0});
  int cos_table =
    cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, cos_table_buf});
  int in = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorRoPE({{in, sin_table, cos_table}, {out}}, circle::RoPEMode_GPT_NEOX);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0, 1.0, 2.0, 3.0}}, {{-1.0, -2.5, 1.0, 3.5}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_RoPE_InvalidShape)
{
  CircleGen cgen;
  uint32_t sin_table_buf = cgen.addBuffer(std::vector<float>{0.5, 1.0, 1.0, 0.5});
  int sin_table =
    cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, sin_table_buf});
  uint32_t cos_table_buf = cgen.addBuffer(std::vector<float>{1.0, 0.5, 0.5, 1.0});
  int cos_table =
    cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, cos_table_buf});
  int in = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1, 3}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorRoPE({{in, sin_table, cos_table}, {out}}, circle::RoPEMode_GPT_NEOX);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailCompile();

  SUCCEED();
}
