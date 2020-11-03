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
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

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
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

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

CircleBuffer genSimpleDepthwiseConv2DQuantizedModel(int input_depth, int depth_multiplier)
{
  assert(1 <= input_depth && input_depth <= 16);
  assert(1 <= depth_multiplier && depth_multiplier <= 16);

  const int output_depth = input_depth * depth_multiplier;
  assert(1 <= output_depth && output_depth <= 16);

  CircleGen cgen;
  uint32_t ker_buf = cgen.addBuffer(
      std::vector<uint8_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1,
                           2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
                           0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});
  uint32_t bias_buf = cgen.addBuffer(std::vector<int32_t>(output_depth, 0));
  int in = cgen.addTensor({{1, 2, 2, input_depth}, circle::TensorType_UINT8}, 0.5, 0);
  int ker = cgen.addTensor({{1, 2, 2, output_depth}, circle::TensorType_UINT8, ker_buf}, 0.5, 0);
  int bias = cgen.addTensor({{output_depth}, circle::TensorType_INT32, bias_buf}, 0.25, 0);
  int out = cgen.addTensor({{1, 1, 1, output_depth}, circle::TensorType_UINT8}, 1, 0);
  cgen.addOperatorDepthwiseConv2D({{in, ker, bias}, {out}}, circle::Padding::Padding_VALID, 1, 1,
                                  depth_multiplier, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});
  return cgen.finish();
}

struct DepthwiseConv2DVariationParam
{
  int input_depth = 0;
  int depth_multiplier = 0;
  std::vector<uint8_t> ref_output;
};

class DepthwiseConv2DVariation : public GenModelTest,
                                 public ::testing::WithParamInterface<DepthwiseConv2DVariationParam>
{
};

static std::vector<uint8_t> input64{
    0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 5, 4, 3, 2, 5, 4, 3, 2, 5, 4, 3, 2, 5, 4, 3, 2,
    2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 4, 6, 8, 2, 3, 5, 8, 8, 5, 3, 2, 1, 2, 3, 4, 5, 4, 3, 2};

TEST_P(DepthwiseConv2DVariation, Test)
{
  // These values must be less than 0 or greater than 2
  auto &param = GetParam();
  _context = std::make_unique<GenModelTestContext>(
      genSimpleDepthwiseConv2DQuantizedModel(param.input_depth, param.depth_multiplier));
  std::vector<uint8_t> ref_input(input64.begin(), input64.begin() + param.input_depth * 4);
  _context->addTestCase(uniformTCD<uint8_t>({ref_input}, {param.ref_output}));
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});

  SUCCEED();
}

// Test with different InputDepth and DepthMultiplier. The values are intended to test optimized CPU
// kernels.
INSTANTIATE_TEST_CASE_P(
    GenModelTest, DepthwiseConv2DVariation,
    ::testing::Values(
        DepthwiseConv2DVariationParam{8, 1, std::vector<uint8_t>{0, 3, 5, 8, 0, 3, 5, 8}},
        DepthwiseConv2DVariationParam{4, 2, std::vector<uint8_t>{0, 0, 2, 3, 0, 2, 6, 9}},
        DepthwiseConv2DVariationParam{
            2, 8, std::vector<uint8_t>{0, 1, 2, 3, 0, 1, 2, 3, 0, 2, 4, 6, 0, 2, 4, 6}},
        DepthwiseConv2DVariationParam{2, 2, std::vector<uint8_t>{0, 1, 4, 6}},
        DepthwiseConv2DVariationParam{2, 1, std::vector<uint8_t>{2, 5}},
        DepthwiseConv2DVariationParam{1, 2, std::vector<uint8_t>{2, 4}},
        DepthwiseConv2DVariationParam{1, 4, std::vector<uint8_t>{0, 2, 3, 5}},
        DepthwiseConv2DVariationParam{4, 1, std::vector<uint8_t>{0, 1, 4, 9}},
        DepthwiseConv2DVariationParam{
            4, 4, std::vector<uint8_t>{0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6, 0, 3, 6, 9}},
        DepthwiseConv2DVariationParam{
            12, 1, std::vector<uint8_t>{0, 3, 7, 12, 0, 4, 7, 12, 0, 4, 9, 16}}));

TEST_F(GenModelTest, neg_OneOp_DepthwiseConv2D_InvalidPaddingType)
{
  _context = std::make_unique<GenModelTestContext>(genNegTestDepthwiseConv2DModel(
      static_cast<circle::Padding>(99), 1, 1, 1, circle::ActivationFunctionType_NONE));
  _context->expectFailModelLoad();
  _context->setBackends({"acl_cl", "acl_neon", "cpu"});
}

// TODO add other invalid operation tests like above
