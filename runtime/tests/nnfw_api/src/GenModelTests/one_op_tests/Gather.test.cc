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

#include "common.h"

TEST_F(GenModelTest, OneOp_Gather_rank5)
{
  CircleGen cgen;

  std::vector<int32_t> index_data{1};

  auto index_buf = cgen.addBuffer(index_data);

  int input = cgen.addTensor({{3, 1, 1, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int indice = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, index_buf});
  int output = cgen.addTensor({{1, 1, 1, 2, 2}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorGather({{input, indice}, {output}}, 0 /*axis*/);
  cgen.setInputsAndOutputs({input}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());

  TestCaseData tc;
  tc.addInput<float>({1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3});
  tc.addOutput<float>({2, 2, 2, 2});
  _context->addTestCase(tc);
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Gather_Q4_0)
{
  CircleGen cgen;

  std::vector<float> params(4 * 32);
  for (uint32_t i = 0; i < params.size(); i++)
  {
    uint32_t sign_bit = i % 2;
    uint32_t multiple = i / 32 + 1;
    uint32_t base = (i / 2) % 8;
    if (sign_bit == 0)
      base += 1;
    params[i] = base * (0.01 * multiple) * (sign_bit ? -1 : 1);
  }

  auto input_vector = quantData(params, circle::TensorType::TensorType_GGML_Q4_0);
  auto input_buf = cgen.addBuffer(input_vector);
  int input = cgen.addTensor({{4, 32}, circle::TensorType::TensorType_GGML_Q4_0, input_buf});
  int indice = cgen.addTensor({{1, 1}, circle::TensorType::TensorType_INT32});
  int output = cgen.addTensor({{1, 32}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorGather({{input, indice}, {output}});
  cgen.setInputsAndOutputs({indice}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());

  TestCaseData tc;
  tc.addInput<int32_t>({2});
  tc.addOutput<float>(std::vector<float>{params.begin() + 64, params.begin() + 96});
  _context->addTestCase(tc);
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Gather_Q4_0_InvalidOutType)
{
  CircleGen cgen;

  std::vector<float> params(4 * 32);

  auto input_vector = quantData(params, circle::TensorType::TensorType_GGML_Q4_0);
  auto input_buf = cgen.addBuffer(input_vector);
  int input = cgen.addTensor({{4, 32}, circle::TensorType::TensorType_GGML_Q4_0, input_buf});
  int indice = cgen.addTensor({{1, 1}, circle::TensorType::TensorType_INT32});
  int output = cgen.addTensor({{1, 32}, circle::TensorType::TensorType_GGML_Q4_0});

  cgen.addOperatorGather({{input, indice}, {output}});
  cgen.setInputsAndOutputs({indice}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Gather_Q4_0_shape)
{
  CircleGen cgen;

  auto input_vector = std::vector<uint8_t>(18);
  auto input_buf = cgen.addBuffer(input_vector);
  int input = cgen.addTensor({{4, 18}, circle::TensorType::TensorType_GGML_Q4_0, input_buf});
  int indice = cgen.addTensor({{1, 1}, circle::TensorType::TensorType_INT32});
  int output = cgen.addTensor({{1, 18}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorGather({{input, indice}, {output}});
  cgen.setInputsAndOutputs({indice}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_Gather_Bool)
{
  CircleGen cgen;

  std::vector<int32_t> index_data{0, 2};
  auto index_buf = cgen.addBuffer(index_data);

  int input = cgen.addTensor({{4}, circle::TensorType::TensorType_BOOL});
  int indice = cgen.addTensor({{2}, circle::TensorType::TensorType_INT32, index_buf});
  int output = cgen.addTensor({{2}, circle::TensorType::TensorType_BOOL});

  cgen.addOperatorGather({{input, indice}, {output}}, 0 /*axis*/);
  cgen.setInputsAndOutputs({input}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());

  TestCaseData tc;
  tc.addInput<bool>({true, false, true, false});
  tc.addOutput<bool>({true, true});
  _context->addTestCase(tc);
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Gather_Bool_InvalidOutputType)
{
  CircleGen cgen;

  std::vector<int32_t> index_data{0};
  auto index_buf = cgen.addBuffer(index_data);

  int input = cgen.addTensor({{4}, circle::TensorType::TensorType_BOOL});
  int indice = cgen.addTensor({{1}, circle::TensorType::TensorType_INT32, index_buf});
  int output = cgen.addTensor({{1}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorGather({{input, indice}, {output}}, 0 /*axis*/);
  cgen.setInputsAndOutputs({input}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}
