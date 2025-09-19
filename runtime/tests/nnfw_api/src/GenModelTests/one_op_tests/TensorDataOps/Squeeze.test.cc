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

TEST_F(GenModelTest, neg_OneOp_Squeeze_invalid_dims)
{
  CircleGen cgen;
  const std::vector<int32_t> squeeze_dims{0, 1}; // 1 dim here is incorrect
  int input = cgen.addTensor({{1, 2, 1, 2}, circle::TensorType::TensorType_FLOAT32});
  int squeeze_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorSqueeze({{input}, {squeeze_out}}, squeeze_dims);
  cgen.setInputsAndOutputs({input}, {squeeze_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"cpu", "gpu_cl"});

  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Squeeze_out_of_rank_dims)
{
  CircleGen cgen;
  const std::vector<int32_t> squeeze_dims{0, 4}; // 4 dim here is incorrect
  int input = cgen.addTensor({{1, 2, 1, 2}, circle::TensorType::TensorType_FLOAT32});
  int squeeze_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorSqueeze({{input}, {squeeze_out}}, squeeze_dims);
  cgen.setInputsAndOutputs({input}, {squeeze_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"cpu", "gpu_cl"});

  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Squeeze_neg_invalid_dims)
{
  CircleGen cgen;
  const std::vector<int32_t> squeeze_dims{0, -3}; // -3 dim here is incorrect
  int input = cgen.addTensor({{1, 2, 1, 2}, circle::TensorType::TensorType_FLOAT32});
  int squeeze_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorSqueeze({{input}, {squeeze_out}}, squeeze_dims);
  cgen.setInputsAndOutputs({input}, {squeeze_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"cpu", "gpu_cl"});

  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Squeeze_neg_out_of_rank_dims)
{
  CircleGen cgen;
  const std::vector<int32_t> squeeze_dims{0, -5}; // -5 dim here is incorrect
  int input = cgen.addTensor({{1, 2, 1, 2}, circle::TensorType::TensorType_FLOAT32});
  int squeeze_out = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorSqueeze({{input}, {squeeze_out}}, squeeze_dims);
  cgen.setInputsAndOutputs({input}, {squeeze_out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"cpu", "gpu_cl"});

  _context->expectFailCompile();

  SUCCEED();
}
