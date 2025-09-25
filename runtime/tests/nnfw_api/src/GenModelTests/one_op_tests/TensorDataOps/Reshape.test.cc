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

TEST_F(GenModelTest, neg_OneOp_Reshape_invalid_target_shape)
{
  CircleGen cgen;
  const auto f32 = circle::TensorType::TensorType_FLOAT32;

  const std::vector<int32_t> new_shape_data{1, 5};
  const uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  const int new_shape = cgen.addTensor({{2}, f32, new_shape_buf});
  const int input = cgen.addTensor({{4}, f32});
  const int out = cgen.addTensor({{1, 5}, f32});

  cgen.addOperatorReshape({{input, new_shape}, {out}}, &new_shape_data);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"cpu", "gpu_cl"});

  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Reshape_invalid_target_dyn_shape)
{
  CircleGen cgen;
  const auto f32 = circle::TensorType::TensorType_FLOAT32;
  const auto i32 = circle::TensorType::TensorType_INT32;

  const std::vector<float> in_data{1.f, 2.f, 3.f, 4.f};
  const uint32_t input_buf = cgen.addBuffer(in_data);
  const int input = cgen.addTensor({{4}, f32, input_buf});
  const int new_shape = cgen.addTensor({{2}, i32});
  const int out = cgen.addTensor({{}, f32}); // unspecified shape

  const CircleGen::Shape empty_new_shape;
  cgen.addOperatorReshape({{input, new_shape}, {out}}, &empty_new_shape);
  cgen.setInputsAndOutputs({new_shape}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput(std::vector<int>{1, 5}).addOutput(in_data).expectFailRun());
  _context->output_sizes(0, sizeof(float) * in_data.size());
  _context->setBackends({"cpu", "gpu_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Reshape_invalid_target_dyn_type)
{
  CircleGen cgen;
  const auto f32 = circle::TensorType::TensorType_FLOAT32;

  const std::vector<float> in_data{1.f, 2.f, 3.f, 4.f};
  const uint32_t input_buf = cgen.addBuffer(in_data);
  const int input = cgen.addTensor({{4}, f32, input_buf});
  const int new_shape = cgen.addTensor({{2}, f32});
  const int out = cgen.addTensor({{}, f32}); // unspecified shape

  const CircleGen::Shape empty_new_shape;
  cgen.addOperatorReshape({{input, new_shape}, {out}}, &empty_new_shape);
  cgen.setInputsAndOutputs({new_shape}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput(std::vector<float>{2, 2}).addOutput(in_data).expectFailRun());
  _context->output_sizes(0, sizeof(float) * in_data.size());
  _context->setBackends({"cpu", "gpu_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Reshape_invalid_target_shape_with_neg_dim)
{
  CircleGen cgen;
  const auto f32 = circle::TensorType::TensorType_FLOAT32;

  const std::vector<int32_t> new_shape_data{5, -1};
  const uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  const int new_shape = cgen.addTensor({{2}, f32, new_shape_buf});
  const int input = cgen.addTensor({{4}, f32});
  const int out = cgen.addTensor({{1, 5}, f32});

  cgen.addOperatorReshape({{input, new_shape}, {out}}, &new_shape_data);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"cpu", "gpu_cl"});

  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Reshape_invalid_target_dyn_shape_with_neg_dim)
{
  CircleGen cgen;
  const auto f32 = circle::TensorType::TensorType_FLOAT32;
  const auto i32 = circle::TensorType::TensorType_INT32;

  const std::vector<float> in_data{1.f, 2.f, 3.f, 4.f};
  const uint32_t input_buf = cgen.addBuffer(in_data);
  const int input = cgen.addTensor({{4}, f32, input_buf});
  const int new_shape = cgen.addTensor({{2}, i32});
  const int out = cgen.addTensor({{}, f32}); // unspecified shape

  const CircleGen::Shape empty_new_shape;
  cgen.addOperatorReshape({{input, new_shape}, {out}}, &empty_new_shape);
  cgen.setInputsAndOutputs({new_shape}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput(std::vector<int>{-1, 5}).addOutput(in_data).expectFailRun());
  _context->output_sizes(0, sizeof(float) * in_data.size());
  _context->setBackends({"cpu", "gpu_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Reshape_invalid_target_shape_with_zero_dim)
{
  CircleGen cgen;
  const auto f32 = circle::TensorType::TensorType_FLOAT32;

  const std::vector<int32_t> new_shape_data{5, 0};
  const uint32_t new_shape_buf = cgen.addBuffer(new_shape_data);
  const int new_shape = cgen.addTensor({{2}, f32, new_shape_buf});
  const int input = cgen.addTensor({{4}, f32});
  const int out = cgen.addTensor({{1, 5}, f32});

  cgen.addOperatorReshape({{input, new_shape}, {out}}, &new_shape_data);
  cgen.setInputsAndOutputs({input}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 3, 4}}, {{1, 2, 3, 4}}));
  _context->setBackends({"cpu", "gpu_cl"});

  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_Reshape_invalid_target_dyn_shape_with_zero_dim)
{
  CircleGen cgen;
  const auto f32 = circle::TensorType::TensorType_FLOAT32;
  const auto i32 = circle::TensorType::TensorType_INT32;

  const std::vector<float> in_data{1.f, 2.f, 3.f, 4.f};
  const uint32_t input_buf = cgen.addBuffer(in_data);
  const int input = cgen.addTensor({{4}, f32, input_buf});
  const int new_shape = cgen.addTensor({{2}, i32});
  const int out = cgen.addTensor({{}, f32}); // unspecified shape

  const CircleGen::Shape empty_new_shape;
  cgen.addOperatorReshape({{input, new_shape}, {out}}, &empty_new_shape);
  cgen.setInputsAndOutputs({new_shape}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    TestCaseData{}.addInput(std::vector<int>{-1, 5}).addOutput(in_data).expectFailRun());
  _context->output_sizes(0, sizeof(float) * in_data.size());
  _context->setBackends({"cpu", "gpu_cl"});

  SUCCEED();
}
