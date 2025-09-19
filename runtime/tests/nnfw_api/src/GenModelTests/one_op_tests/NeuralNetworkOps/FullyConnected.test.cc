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

TEST_F(GenModelTest, OneOp_FullyConnected)
{
  CircleGen cgen;
  // clang-format off
  std::vector<float> weight_data{ 1, 0, 0, 1,
                                  2, 0, 0, -1,
                                  3, 0, 0, 2,
                                  4, 0, 0, 1,
                                  1, 0, 0, 1,
                                  2, 0, 0, -1,
                                  3, 0, 0, 2,
                                  4, 0, 0, 1,
                                  1, 0, 0, 1,
                                  2, 0, 0, -1,
                                  3, 0, 0, 2,
                                  4, 0, 0, 1,
                                  1, 0, 0, 1,
                                  2, 0, 0, -1,
                                  3, 0, 0, 2,
                                  4, 0, 0, 1 };
  std::vector<float> bias_data{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
  // clang-format on
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int input = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{16, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{16}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int output = cgen.addTensor({{1, 16}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight, bias}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 3, 2, 1}}, {{2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 6}}));
  _context->setBackends({"cpu", "acl_neon", "xnnpack", "ruy"});

  SUCCEED();
}

#if defined(__aarch64__)
TEST_F(GenModelTest, OneOp_FullyConnectedShuffled16x1Float32)
{
  CircleGen cgen;
  // clang-format off
  std::vector<float> weight_data{ 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  1, -1, 2, 1, 1, -1, 2, 1, 1, -1, 2, 1, 1, -1, 2, 1 };
  std::vector<float> bias_data{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
  // clang-format on
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int input = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{16, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{16}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int output = cgen.addTensor({{1, 16}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight, bias}, {output}},
                                 circle::FullyConnectedOptionsWeightsFormat_SHUFFLED16x1FLOAT32);
  cgen.setInputsAndOutputs({input}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 3, 2, 1}}, {{2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 6}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}
#endif

// Failure is expected except for aarch64 and cpu backend
TEST_F(GenModelTest, OneOp_neg_FullyConnectedShuffled16x1Float32)
{
  CircleGen cgen;
  // clang-format off
  std::vector<float> weight_data{ 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                  1, -1, 2, 1, 1, -1, 2, 1, 1, -1, 2, 1, 1, -1, 2, 1,
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  std::vector<float> bias_data{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
  // clang-format on
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int input = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{16, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{16}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int output = cgen.addTensor({{1, 16}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight, bias}, {output}},
                                 circle::FullyConnectedOptionsWeightsFormat_SHUFFLED16x1FLOAT32);
  cgen.setInputsAndOutputs({input}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  auto tc = uniformTCD<float>({{1, 3, 2, 1}}, {{2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 6}});
  _context->addTestCase(tc);
  _context->setBackends({"acl_neon", "acl_cl"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_FullyConnected16x1Sparse)
{
  CircleGen cgen;
  // clang-format off
  std::vector<float> weight_data{ 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                  1, -1, 2, 1, 1, -1, 2, 1, 1, -1, 2, 1, 1, -1, 2, 1};
  std::vector<float> bias_data{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 };
  // clang-format on
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int input = cgen.addTensor({{1, 4}, circle::TensorType::TensorType_FLOAT32});
  CircleGen::SparsityParams sp{
    {0, 1, 2, 3},
    {0, 1},
    {{CircleGen::SparseDimensionType::DimensionType_DENSE, 1},
     {CircleGen::SparseDimensionType::DimensionType_SPARSE_CSR, {0, 2}, {0, 3}},
     {CircleGen::SparseDimensionType::DimensionType_DENSE, 16},
     {CircleGen::SparseDimensionType::DimensionType_DENSE, 1}}};
  int weight = cgen.addTensor({{16, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf}, sp);
  int bias = cgen.addTensor({{16}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int output = cgen.addTensor({{1, 16}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight, bias}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{1, 3, 2, 1}}, {{2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 5, 2, 1, 5, 6}}));
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_FullyConnected_OptionalBias)
{
  CircleGen cgen;
  // clang-format off
  std::vector<float> weight_data{ -1,  4,  0,  3,
                                   1,  4,  0, -1,
                                   3, -1,  0, -1,
                                  -1,  3,  4,  4,
                                   4,  0,  4,  0,
                                   4,  1, -1,  1,
                                   2,  2, -2, -1,
                                   4, -1, -2,  3 };
  // clang-format on
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  int input = cgen.addTensor({{2, 4}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{8, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int output = cgen.addTensor({{2, 8}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight, -1 /* Optional bias */}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{3, -1, -1, 1, -2, 0, -2, 1}},
                      {{-4, -2, 9, -6, 8, 13, 5, 18, 5, -3, -7, -2, -16, -5, -1, -1}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "xnnpack", "ruy"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_FullyConnected_NoBias)
{
  CircleGen cgen;
  // clang-format off
  std::vector<float> weight_data{ -1,  4,  0,  3,
                                   1,  4,  0, -1,
                                   3, -1,  0, -1,
                                  -1,  3,  4,  4,
                                   4,  0,  4,  0,
                                   4,  1, -1,  1,
                                   2,  2, -2, -1,
                                   4, -1, -2,  3 };
  // clang-format on
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  int input = cgen.addTensor({{2, 4}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{8, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int output = cgen.addTensor({{2, 8}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorFullyConnected({{input, weight /* Missing bias */}, {output}});
  cgen.setInputsAndOutputs({input}, {output});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{3, -1, -1, 1, -2, 0, -2, 1}},
                      {{-4, -2, 9, -6, 8, 13, 5, 18, 5, -3, -7, -2, -16, -5, -1, -1}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "xnnpack", "ruy"});
  _context->expectFailCompile();

  SUCCEED();
}
