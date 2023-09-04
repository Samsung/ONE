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

TEST_F(GenModelTest, OneOp_DepthwiseConv2D)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{1, 2, 3, 4};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12}},
                                          {{71, -34, 99, -20, 91, -26, 127, -4}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "xnnpack"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_DepthwiseConv2D_No_Multiplier)
{
  CircleGen cgen;
  std::vector<float> weight_data{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{0.5f, -0.5f};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 3, 1, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_SAME, 1, 1, 1,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
                      {{16.5f, 27.5f, 28.5f, 43.5f, 8.5f, 15.5f, 12.5f, 23.5f}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "gpu_cl"});
  SUCCEED();
}

TEST_F(GenModelTest, OneOp_DepthwiseConv2D_No_Multiplier_RELU6)
{
  CircleGen cgen;
  std::vector<float> weight_data{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{0.5f, -0.5f};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 3, 1, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_SAME, 1, 1, 1,
                                  circle::ActivationFunctionType_RELU6);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
                                          {{6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f, 6.0f}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "gpu_cl"});
  SUCCEED();
}

TEST_F(GenModelTest, OneOp_DepthwiseConv2D_3x3)
{
  CircleGen cgen;
  std::vector<float> weight_data{0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{0.0f, 0.0f};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_SAME, 1, 1, 1,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(
    uniformTCD<float>({{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}},
                      {{6.0f, 16.0f, 8.0f, 16.0f, 10.0f, 16.0f, 12.0f, 16.0f}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "gpu_cl"});
  SUCCEED();
}

TEST_F(GenModelTest, OneOp_DepthwiseConv2D_Dilation)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{0, 0, 0, 0};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 4, 4, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE, 2, 2);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{
                                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                                            0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          }},
                                          {{13, 14, 0, 0, 0, 0, 11, 12, 5, 6, 0, 0, 0, 0, 3, 4}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "xnnpack"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_DepthwiseConv2D_Dilation_N_Stride)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{0, 0, 0, 0};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 6, 6, 1}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 1}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 1}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 3, 3, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_SAME, 2, 2, 1,
                                  circle::ActivationFunctionType_NONE, 3, 3);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(uniformTCD<float>({{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                                            0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
                                          {{4, 0, 3, 0, 0, 0, 2, 0, 1}}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "xnnpack", "gpu_cl"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_DepthwiseConv2D_U8_PerChannel)
{
  CircleGen cgen;
  // weight
  // clang-format off
  std::vector<uint8_t> weight_data{2, 1, 2,
                                   6, 2, 3,
                                   2, 3, 4,
                                   4, 4, 5};
  // clang-format on
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> weight_scales = {.5, 1, 2};
  std::vector<int64_t> weight_zeropoints = {2, 0, 1};
  int weight = cgen.addTensor({{1, 2, 2, 3}, circle::TensorType::TensorType_UINT8, weight_buf},
                              weight_scales, weight_zeropoints);
  // bias
  std::vector<int32_t> bias_data{4, -8, -4};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int bias = cgen.addTensor({{1, 1, 1, 3}, circle::TensorType::TensorType_INT32, bias_buf}, 1., 0);

  // in and out
  int in = cgen.addTensor({{1, 2, 2, 3}, circle::TensorType::TensorType_UINT8}, 2., 1);
  int out = cgen.addTensor({{1, 1, 1, 3}, circle::TensorType::TensorType_UINT8}, 4., 2);

  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 1,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  // clang-format off
  _context->addTestCase(uniformTCD<uint8_t>({{5, 5, 5,  // NHWC
                                              3, 3, 3,
                                              7, 7, 7,
                                              9, 9, 9}
                                            },
                                            {{9,
                                              27,
                                              56}
                                            }));
  // clang-format on
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, OneOp_DepthwiseConv2D_I8_Hybrid_PerChannel)
{
  CircleGen cgen;
  // weight
  // clang-format off
  std::vector<int8_t> weight_data{1, 2, 1, 2,      -9,  10, -9,  10,
                                  5, 6, 5, 6,      13, -14, 13, -14};
  // clang-format on
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> weight_scales = {1, 1, 1, 1};
  std::vector<int64_t> weight_zeropoints = {0, 0, 0, 0};
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_INT8, weight_buf},
                              weight_scales, weight_zeropoints);
  // bias
  std::vector<float> bias_data{0, 1, 2, 3};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int bias = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, bias_buf});

  // in and out
  int in = cgen.addTensor({{1, 3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_FLOAT32});

  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  // clang-format off
  _context->addTestCase(uniformTCD<float>({{0, 1,     2, 3,
                                            0, 1,     2, 3,
                                            0, 1,     2, 3}},
                                          {{8, -7, 20, -1,
                                            8, -7, 20, -1}}));
  // clang-format on
  _context->setBackends({"cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_DepthwiseConv2D_Stride)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{1, 2, 3, 4};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 0, 0, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_DepthwiseConv2D_Dilation)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{0, 0, 0, 0};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 4, 4, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE, 0, 0);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_DepthwiseConv2D_Type)
{
  CircleGen cgen;
  std::vector<float> weight_data{1, 2, 3, 4, -9, 10, -11, 12, 5, 6, 7, 8, 13, -14, 15, -16};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{1, 2, 3, 4};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 3, 2, 2}, circle::TensorType::TensorType_FLOAT32});
  int weight = cgen.addTensor({{1, 2, 2, 4}, circle::TensorType::TensorType_FLOAT32, weight_buf});
  int bias = cgen.addTensor({{1, 1, 1, 4}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 1, 4}, circle::TensorType::TensorType_UINT8});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

// Generate a model for negative test cases
CircleBuffer genNegTestDepthwiseConv2DModel(circle::Padding padding, int stride_w, int stride_h,
                                            int depth_multiplier,
                                            circle::ActivationFunctionType actfn)
{
  CircleGen cgen;
  uint32_t ker_buf = cgen.addBuffer(std::vector<uint8_t>{0, 1, 2, 3, 0, 1, 2, 3});
  uint32_t bias_buf = cgen.addBuffer(std::vector<int32_t>{0, 0});
  int in = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType_UINT8}, 0.5, 0);
  int ker = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType_UINT8, ker_buf}, 0.5, 0);
  int bias = cgen.addTensor({{2}, circle::TensorType_INT32, bias_buf}, 0.25, 0);
  int out = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType_UINT8}, 1, 0);
  cgen.addOperatorDepthwiseConv2D({{in, ker, bias}, {out}}, padding, stride_w, stride_h,
                                  depth_multiplier, actfn, 0, 0);
  cgen.setInputsAndOutputs({in}, {out});
  return cgen.finish();
}

template <typename T> struct DepthwiseConv2DQuantTestParam
{
  int stride = 1; // Used for both height and width
  int input_depth = 1;
  int depth_multiplier = 1;
  std::vector<T> ref_output;
};

template <typename T>
class DepthwiseConv2DQuantTest
  : public GenModelTest,
    public ::testing::WithParamInterface<DepthwiseConv2DQuantTestParam<T>>
{
};

using DepthwiseConv2DQuantTestParamU8 = DepthwiseConv2DQuantTestParam<uint8_t>;
using DepthwiseConv2DQuantTestU8 = DepthwiseConv2DQuantTest<uint8_t>;

// Test with different InputDepth and DepthMultiplier. The values are intended to test optimized CPU
// kernels.
INSTANTIATE_TEST_SUITE_P(
  GenModelTest, DepthwiseConv2DQuantTestU8,
  ::testing::Values(
    // Stride == 1
    DepthwiseConv2DQuantTestParamU8{1, 8, 1, std::vector<uint8_t>{0, 3, 5, 8, 0, 3, 5, 8}},
    DepthwiseConv2DQuantTestParamU8{1, 4, 2, std::vector<uint8_t>{0, 0, 2, 3, 0, 2, 6, 9}},
    DepthwiseConv2DQuantTestParamU8{
      1, 2, 8, std::vector<uint8_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 4, 6, 0, 2, 4, 6}},
    DepthwiseConv2DQuantTestParamU8{1, 2, 2, std::vector<uint8_t>{0, 1, 4, 6}},
    DepthwiseConv2DQuantTestParamU8{1, 2, 1, std::vector<uint8_t>{2, 5}},
    DepthwiseConv2DQuantTestParamU8{1, 1, 2, std::vector<uint8_t>{2, 4}},
    DepthwiseConv2DQuantTestParamU8{1, 1, 4, std::vector<uint8_t>{0, 2, 3, 5}},
    DepthwiseConv2DQuantTestParamU8{1, 4, 1, std::vector<uint8_t>{0, 1, 4, 9}},
    DepthwiseConv2DQuantTestParamU8{
      1, 4, 4, std::vector<uint8_t>{0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9}},
    DepthwiseConv2DQuantTestParamU8{1, 12, 1,
                                    std::vector<uint8_t>{0, 3, 7, 12, 0, 4, 7, 12, 0, 4, 9, 16}},
    // Stride == 2
    DepthwiseConv2DQuantTestParamU8{2, 4, 1, std::vector<uint8_t>{0, 1, 4, 9}},
    DepthwiseConv2DQuantTestParamU8{2, 2, 1, std::vector<uint8_t>{2, 5}},
    DepthwiseConv2DQuantTestParamU8{2, 1, 8, std::vector<uint8_t>{0, 2, 3, 5, 0, 2, 3, 5}},
    DepthwiseConv2DQuantTestParamU8{2, 1, 32, std::vector<uint8_t>{0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3,
                                                                   5, 0, 2, 3, 5, 0, 2, 3, 5, 0, 2,
                                                                   3, 5, 0, 2, 3, 5, 0, 2, 3, 5}},
    DepthwiseConv2DQuantTestParamU8{
      2, 1, 20, std::vector<uint8_t>{0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3, 5}},
    DepthwiseConv2DQuantTestParamU8{
      2, 1, 16, std::vector<uint8_t>{0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3, 5}},
    DepthwiseConv2DQuantTestParamU8{2, 8, 1, std::vector<uint8_t>{0, 3, 5, 8, 0, 3, 5, 8}},
    DepthwiseConv2DQuantTestParamU8{
      2, 8, 2, std::vector<uint8_t>{0, 3, 5, 8, 0, 3, 5, 8, 0, 3, 5, 8, 0, 3, 5, 8}},
    DepthwiseConv2DQuantTestParamU8{
      2, 16, 1, std::vector<uint8_t>{0, 3, 8, 16, 0, 4, 7, 12, 0, 3, 7, 13, 0, 4, 7, 12}}));

CircleBuffer genDepthwiseConv2DQuantU8Model(int stride, int input_depth, int depth_multiplier)
{
  assert(1 <= stride && stride <= 2);
  assert(1 <= input_depth && input_depth <= 16);
  assert(1 <= depth_multiplier && depth_multiplier <= 32);

  const int output_depth = input_depth * depth_multiplier;
  assert(1 <= output_depth && output_depth <= 32);

  CircleGen cgen;
  uint32_t ker_buf = cgen.addBuffer(std::vector<uint8_t>{
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
    2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
    2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});
  uint32_t bias_buf = cgen.addBuffer(std::vector<int32_t>(output_depth, 0));
  int in = cgen.addTensor({{1, 2, 2, input_depth}, circle::TensorType_UINT8}, 0.5, 0);
  int ker = cgen.addTensor({{1, 2, 2, output_depth}, circle::TensorType_UINT8, ker_buf}, 0.5, 0);
  int bias = cgen.addTensor({{output_depth}, circle::TensorType_INT32, bias_buf}, 0.25, 0);
  int out = cgen.addTensor({{1, 1, 1, output_depth}, circle::TensorType_UINT8}, 1, 0);
  cgen.addOperatorDepthwiseConv2D({{in, ker, bias}, {out}}, circle::Padding::Padding_VALID, stride,
                                  stride, depth_multiplier, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});
  return cgen.finish();
}

TEST_P(DepthwiseConv2DQuantTestU8, Test)
{
  // Same input is used for all tests but output differs
  static const std::vector<uint8_t> input64{
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 5, 4, 3, 2, 5, 4, 3, 2, 5, 4, 3, 2, 5, 4, 3, 2,
    2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 3, 5, 8, 8, 5, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2};

  auto &param = GetParam();
  _context = std::make_unique<GenModelTestContext>(
    genDepthwiseConv2DQuantU8Model(param.stride, param.input_depth, param.depth_multiplier));
  std::vector<uint8_t> ref_input(input64.begin(), input64.begin() + param.input_depth * 4);
  _context->addTestCase(uniformTCD<uint8_t>({ref_input}, {param.ref_output}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

using DepthwiseConv2DQuantTestParamI8 = DepthwiseConv2DQuantTestParam<int8_t>;
using DepthwiseConv2DQuantTestI8 = DepthwiseConv2DQuantTest<int8_t>;

// Test with different InputDepth and DepthMultiplier. The values are intended to test optimized CPU
// kernels.
INSTANTIATE_TEST_SUITE_P(
  GenModelTest, DepthwiseConv2DQuantTestI8,
  ::testing::Values(
    // Stride == 1
    DepthwiseConv2DQuantTestParamI8{1, 8, 1, std::vector<int8_t>{0, 3, 5, 8, 0, 3, 5, 8}},
    DepthwiseConv2DQuantTestParamI8{1, 4, 2, std::vector<int8_t>{0, 0, 2, 3, 0, 2, 6, 9}},
    DepthwiseConv2DQuantTestParamI8{
      1, 2, 8, std::vector<int8_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 4, 6, 0, 2, 4, 6}},
    DepthwiseConv2DQuantTestParamI8{1, 2, 2, std::vector<int8_t>{0, 1, 4, 6}},
    DepthwiseConv2DQuantTestParamI8{1, 2, 1, std::vector<int8_t>{2, 5}},
    DepthwiseConv2DQuantTestParamI8{1, 1, 2, std::vector<int8_t>{2, 4}},
    DepthwiseConv2DQuantTestParamI8{1, 1, 4, std::vector<int8_t>{0, 2, 3, 5}},
    DepthwiseConv2DQuantTestParamI8{1, 4, 1, std::vector<int8_t>{0, 1, 4, 9}},
    DepthwiseConv2DQuantTestParamI8{
      1, 4, 4, std::vector<int8_t>{0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9}},
    DepthwiseConv2DQuantTestParamI8{1, 12, 1,
                                    std::vector<int8_t>{0, 3, 7, 12, 0, 4, 7, 12, 0, 4, 9, 16}},
    // Stride == 2
    DepthwiseConv2DQuantTestParamI8{2, 4, 1, std::vector<int8_t>{0, 1, 4, 9}},
    DepthwiseConv2DQuantTestParamI8{2, 2, 1, std::vector<int8_t>{2, 5}},
    DepthwiseConv2DQuantTestParamI8{2, 1, 8, std::vector<int8_t>{0, 2, 3, 5, 0, 2, 3, 5}},
    DepthwiseConv2DQuantTestParamI8{2, 1, 32, std::vector<int8_t>{0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3,
                                                                  5, 0, 2, 3, 5, 0, 2, 3, 5, 0, 2,
                                                                  3, 5, 0, 2, 3, 5, 0, 2, 3, 5}},
    DepthwiseConv2DQuantTestParamI8{
      2, 1, 20, std::vector<int8_t>{0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3, 5}},
    DepthwiseConv2DQuantTestParamI8{
      2, 1, 16, std::vector<int8_t>{0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3, 5, 0, 2, 3, 5}},
    DepthwiseConv2DQuantTestParamI8{2, 8, 1, std::vector<int8_t>{0, 3, 5, 8, 0, 3, 5, 8}},
    DepthwiseConv2DQuantTestParamI8{
      2, 8, 2, std::vector<int8_t>{0, 3, 5, 8, 0, 3, 5, 8, 0, 3, 5, 8, 0, 3, 5, 8}},
    DepthwiseConv2DQuantTestParamI8{
      2, 16, 1, std::vector<int8_t>{0, 3, 8, 16, 0, 4, 7, 12, 0, 3, 7, 13, 0, 4, 7, 12}}));

CircleBuffer genDepthwiseConv2DQuantI8Model(int stride, int input_depth, int depth_multiplier)
{
  assert(1 <= stride && stride <= 2);
  assert(1 <= input_depth && input_depth <= 16);
  assert(1 <= depth_multiplier && depth_multiplier <= 32);

  const int output_depth = input_depth * depth_multiplier;
  assert(1 <= output_depth && output_depth <= 32);

  CircleGen cgen;
  uint32_t ker_buf = cgen.addBuffer(std::vector<int8_t>{
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
    2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
    2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});
  uint32_t bias_buf = cgen.addBuffer(std::vector<int32_t>(output_depth, 0));
  int in = cgen.addTensor({{1, 2, 2, input_depth}, circle::TensorType_INT8}, 0.5, 0);
  int ker = cgen.addTensor({{1, 2, 2, output_depth}, circle::TensorType_INT8, ker_buf}, 0.5, 0);
  int bias = cgen.addTensor({{output_depth}, circle::TensorType_INT32, bias_buf}, 0.25, 0);
  int out = cgen.addTensor({{1, 1, 1, output_depth}, circle::TensorType_INT8}, 1, 0);
  cgen.addOperatorDepthwiseConv2D({{in, ker, bias}, {out}}, circle::Padding::Padding_VALID, stride,
                                  stride, depth_multiplier, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});
  return cgen.finish();
}

TEST_P(DepthwiseConv2DQuantTestI8, Test)
{
  // Same input is used for all tests but output differs
  static const std::vector<int8_t> input64{
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 5, 4, 3, 2, 5, 4, 3, 2, 5, 4, 3, 2, 5, 4, 3, 2,
    2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 3, 5, 8, 8, 5, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2};

  auto &param = GetParam();
  _context = std::make_unique<GenModelTestContext>(
    genDepthwiseConv2DQuantI8Model(param.stride, param.input_depth, param.depth_multiplier));
  std::vector<int8_t> ref_input(input64.begin(), input64.begin() + param.input_depth * 4);
  _context->addTestCase(uniformTCD<int8_t>({ref_input}, {param.ref_output}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_DepthwiseConv2D_InvalidPaddingType)
{
  _context = std::make_unique<GenModelTestContext>(genNegTestDepthwiseConv2DModel(
    static_cast<circle::Padding>(99), 1, 1, 1, circle::ActivationFunctionType_NONE));
  _context->expectFailModelLoad();
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "xnnpack"});

  SUCCEED();
}

// TODO add other invalid operation tests like above

TEST_F(GenModelTest, neg_OneOp_DepthwiseConv2D_I8_NonZero_ZeroPoints)
{
  CircleGen cgen;
  std::vector<int8_t> weight_data{1, 2, 3, 4, 5, 6, 7, 8};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<int32_t> bias_data{0, 2};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 3, 3, 2}, circle::TensorType::TensorType_INT8}, 0.5, 0);
  std::vector<float> weight_scales = {0.5, 1};
  std::vector<int64_t> weight_zeropoints = {0, 10};
  int weight = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_INT8, weight_buf},
                              weight_scales, weight_zeropoints);
  int bias = cgen.addTensor({{1, 1, 1, 2}, circle::TensorType::TensorType_INT32, bias_buf});
  int out = cgen.addTensor({{1, 2, 2, 2}, circle::TensorType::TensorType_FLOAT32}, 1.0, 0);
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1, 2,
                                  circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});
  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"cpu"});
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_DepthwiseConv2D_I8_Hybrid_PerTensor)
{
  // PerTensor Quantized Weight is not supported
  CircleGen cgen;
  std::vector<int8_t> weight_data{1, 2, 3};
  uint32_t weight_buf = cgen.addBuffer(weight_data);
  std::vector<float> bias_data{0, 2, 4};
  uint32_t bias_buf = cgen.addBuffer(bias_data);
  int in = cgen.addTensor({{1, 1, 1, 3}, circle::TensorType::TensorType_FLOAT32});
  // Hybrid does not support per-tensor.
  std::vector<float> weight_scales = {0.5};
  std::vector<int64_t> weight_zeropoints = {0};
  int weight = cgen.addTensor({{1, 1, 1, 3}, circle::TensorType::TensorType_INT8, weight_buf},
                              weight_scales, weight_zeropoints);
  int bias = cgen.addTensor({{1, 1, 1, 3}, circle::TensorType::TensorType_FLOAT32, bias_buf});
  int out = cgen.addTensor({{1, 1, 1, 3}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorDepthwiseConv2D({{in, weight, bias}, {out}}, circle::Padding_VALID, 1, 1,
                                  /* depth_multiplier */ 1, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailCompile();
  _context->setBackends({"cpu"});
  SUCCEED();
}
