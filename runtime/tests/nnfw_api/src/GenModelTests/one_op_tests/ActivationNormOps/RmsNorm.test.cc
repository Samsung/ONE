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

TEST_F(GenModelTest, OneOp_RmsNorm)
{
  CircleGen cgen;
  uint32_t gamma_buf = cgen.addBuffer(std::vector<float>{1});
  int gamma = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, gamma_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorRmsNorm({{in, gamma}, {out}}, 0.00001f);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0, 1, 2, 3}}, {{0, 1, 1, 1}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_RmsNorm_InvalidShape)
{
  CircleGen cgen;
  uint32_t gamma_buf = cgen.addBuffer(std::vector<float>{2});
  int gamma = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32, gamma_buf});
  int in = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{2, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorRmsNorm({{in, gamma}, {out}}, 0.00001f);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailCompile();

  SUCCEED();
}
