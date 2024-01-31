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

struct AvgPool2DParam
{
  TestCaseData tcd;
  std::vector<int32_t> input_shape;
  std::vector<int32_t> output_shape;
  struct filter_stride
  {
    int32_t filter_w;
    int32_t filter_h;
    int32_t stride_w;
    int32_t stride_h;
  } param = {1, 1, 1, 1};
  struct data_type
  {
    circle::TensorType data_type;
    float scale;
    int64_t zero_point;
  } type = {circle::TensorType::TensorType_FLOAT32, 0.0f, 0};
  std::vector<std::string> backend = {"acl_cl", "acl_neon", "cpu", "gpu_cl"};
};

class AveragePool2DVariation : public GenModelTest,
                               public ::testing::WithParamInterface<AvgPool2DParam>
{
};

// Test with different input type and value
INSTANTIATE_TEST_SUITE_P(
  GenModelTest, AveragePool2DVariation,
  ::testing::Values(
    // float data
    AvgPool2DParam{
      uniformTCD<float>({{1, 3, 2, 4}}, {{2.5}}), {1, 2, 2, 1}, {1, 1, 1, 1}, {2, 2, 2, 2}},
    // float data - large
    AvgPool2DParam{uniformTCD<float>({std::vector<float>(18 * 36 * 2, 99)}, {{99, 99, 99, 99}}),
                   {1, 18, 36, 2},
                   {1, 1, 2, 2},
                   {18, 18, 18, 18}},
    // uint8_t data
    AvgPool2DParam{uniformTCD<uint8_t>({{2, 6, 4, 8}}, {{5}}),
                   {1, 2, 2, 1},
                   {1, 1, 1, 1},
                   {2, 2, 2, 2},
                   {circle::TensorType::TensorType_UINT8, 1.2, 3},
                   {"acl_cl", "acl_neon", "cpu"}},
    // uint8_t data -large
    AvgPool2DParam{
      uniformTCD<uint8_t>({{std::vector<uint8_t>(18 * 36 * 2, 99)}}, {{99, 99, 99, 99}}),
      {1, 18, 36, 2},
      {1, 1, 2, 2},
      {18, 18, 18, 18},
      {circle::TensorType::TensorType_UINT8, 1.2, 3},
      {"acl_cl", "acl_neon", "cpu"}},
    // int8_t data
    // TODO enable acl-cl, acl-neon backend
    AvgPool2DParam{uniformTCD<int8_t>({{2, -6, 4, -8}}, {{-2}}),
                   {1, 2, 2, 1},
                   {1, 1, 1, 1},
                   {2, 2, 2, 2},
                   {circle::TensorType::TensorType_INT8, 2.0, -1},
                   {"cpu"}},
    // int8_t data - large
    // TODO enable acl-cl, acl-neon backend
    AvgPool2DParam{
      uniformTCD<int8_t>({{std::vector<int8_t>(18 * 36 * 2, -99)}}, {{-99, -99, -99, -99}}),
      {1, 18, 36, 2},
      {1, 1, 2, 2},
      {18, 18, 18, 18},
      {circle::TensorType::TensorType_INT8, 2.0, -1},
      {"cpu"}}));

TEST_P(AveragePool2DVariation, Test)
{
  auto &param = GetParam();
  CircleGen cgen;

  int in = cgen.addTensor({param.input_shape, param.type.data_type}, param.type.scale,
                          param.type.zero_point);
  int out = cgen.addTensor({param.output_shape, param.type.data_type}, param.type.scale,
                           param.type.zero_point);
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, param.param.stride_w,
                                param.param.stride_h, param.param.filter_w, param.param.filter_h,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->addTestCase(param.tcd);
  _context->setBackends(param.backend);

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_AvgPool2D_3DInput)
{
  // 3D Tensors are not supported
  CircleGen cgen;
  int in = cgen.addTensor({{2, 2, 1}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 2, 2, 2, 2,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "gpu_cl"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_F(GenModelTest, neg_OneOp_AvgPool2D_2DInput)
{
  // 2D Tensors are not supported
  CircleGen cgen;
  int in = cgen.addTensor({{2, 2}, circle::TensorType::TensorType_FLOAT32});
  int out = cgen.addTensor({{1, 1}, circle::TensorType::TensorType_FLOAT32});
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 2, 2, 2, 2,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->setBackends({"acl_cl", "acl_neon", "cpu", "gpu_cl"});
  _context->expectFailCompile();

  SUCCEED();
}

TEST_P(AveragePool2DVariation, neg_InvalidPaddingType)
{
  auto &param = GetParam();
  CircleGen cgen;

  int in = cgen.addTensor({param.input_shape, param.type.data_type}, param.type.scale,
                          param.type.zero_point);
  int out = cgen.addTensor({param.output_shape, param.type.data_type}, param.type.scale,
                           param.type.zero_point);
  cgen.addOperatorAveragePool2D({{in}, {out}}, static_cast<circle::Padding>(99),
                                param.param.stride_w, param.param.stride_h, param.param.filter_w,
                                param.param.filter_h, circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_P(AveragePool2DVariation, neg_InvalidFilterSize_1)
{
  auto &param = GetParam();
  CircleGen cgen;

  int in = cgen.addTensor({param.input_shape, param.type.data_type}, param.type.scale,
                          param.type.zero_point);
  int out = cgen.addTensor({param.output_shape, param.type.data_type}, param.type.scale,
                           param.type.zero_point);
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, param.param.stride_w,
                                param.param.stride_h, -1, param.param.filter_h,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_P(AveragePool2DVariation, neg_InvalidFilterSize_2)
{
  auto &param = GetParam();
  CircleGen cgen;

  int in = cgen.addTensor({param.input_shape, param.type.data_type}, param.type.scale,
                          param.type.zero_point);
  int out = cgen.addTensor({param.output_shape, param.type.data_type}, param.type.scale,
                           param.type.zero_point);
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, param.param.stride_w,
                                param.param.stride_h, param.param.filter_w, 0,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_P(AveragePool2DVariation, neg_InvalidStrides_1)
{
  auto &param = GetParam();
  CircleGen cgen;

  int in = cgen.addTensor({param.input_shape, param.type.data_type}, param.type.scale,
                          param.type.zero_point);
  int out = cgen.addTensor({param.output_shape, param.type.data_type}, param.type.scale,
                           param.type.zero_point);
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, 0, param.param.stride_h,
                                param.param.filter_w, param.param.filter_h,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}

TEST_P(AveragePool2DVariation, neg_InvalidStrides_2)
{
  auto &param = GetParam();
  CircleGen cgen;

  int in = cgen.addTensor({param.input_shape, param.type.data_type}, param.type.scale,
                          param.type.zero_point);
  int out = cgen.addTensor({param.output_shape, param.type.data_type}, param.type.scale,
                           param.type.zero_point);
  cgen.addOperatorAveragePool2D({{in}, {out}}, circle::Padding_SAME, param.param.stride_w, -100,
                                param.param.filter_w, param.param.filter_h,
                                circle::ActivationFunctionType_NONE);
  cgen.setInputsAndOutputs({in}, {out});

  _context = std::make_unique<GenModelTestContext>(cgen.finish());
  _context->expectFailModelLoad();

  SUCCEED();
}
