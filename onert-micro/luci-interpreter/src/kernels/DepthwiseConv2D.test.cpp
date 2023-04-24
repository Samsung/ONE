/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels/DepthwiseConv2D.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class DepthwiseConv2DTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(DepthwiseConv2DTest, Float)
{
  Shape input_shape{1, 4, 2, 2};
  Shape filter_shape{1, 2, 2, 4};
  Shape bias_shape{4};
  std::vector<float> input_data{
    1,  2,  7,  8,  //
    3,  4,  9,  10, //
    5,  6,  11, 12, //
    13, 14, 15, 16, //
  };
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  std::vector<float> bias_data{1, 2, 3, 4};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor scratchpad(DataType::FLOAT32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  DepthwiseConv2DParams params{};
  params.padding = Padding::VALID;
  params.depth_multiplier = 2;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  DepthwiseConv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &scratchpad,
                         params);
  kernel.configure();
  _memory_manager->allocate_memory(scratchpad);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{
    71,  0, 99,  0,  //
    167, 0, 227, 28, //
  };
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 1, 4}));
}

TEST_F(DepthwiseConv2DTest, Uint8)
{
  std::vector<float> input_data{
    1, 2, 7,  8,  // column 1
    3, 4, 9,  10, // column 2
    5, 6, 11, 12, // column 3
  };
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  std::vector<float> bias_data{1, 2, 3, 4};

  std::pair<float, int32_t> input_quant_param = quantizationParams<uint8_t>(-63.5, 64);
  std::pair<float, int32_t> output_quant_param = quantizationParams<uint8_t>(-127, 128);

  Tensor input_tensor =
    makeInputTensor<DataType::U8>({1, 3, 2, 2}, input_quant_param.first, input_quant_param.second,
                                  input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::U8>({1, 2, 2, 4}, input_quant_param.first, input_quant_param.second,
                                  filter_data, _memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S32>(
    {4}, input_quant_param.first * input_quant_param.first, 0, bias_data, _memory_manager.get());
  Tensor output_tensor =
    makeOutputTensor(DataType::U8, output_quant_param.first, output_quant_param.second);
  Tensor scratchpad(DataType::FLOAT32, Shape({}), {}, "");

  DepthwiseConv2DParams params{};
  params.padding = Padding::VALID;
  params.depth_multiplier = 2;
  params.stride_height = 1;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::NONE;

  DepthwiseConv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &scratchpad,
                         params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(scratchpad);
  kernel.execute();

  std::vector<float> ref_output_data{
    71, -34, 99,  -20, //
    91, -26, 127, -4,  //
  };
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 1, 4}));
}

TEST_F(DepthwiseConv2DTest, SInt16)
{
  Shape input_shape{1, 4, 2, 2};
  Shape filter_shape{1, 2, 2, 4};
  Shape bias_shape{4};
  std::vector<int32_t> ref_output_shape{1, 2, 1, 4};

  std::vector<float> input_data{
    1,  2,  7,  8,  //
    3,  4,  9,  10, //
    5,  6,  11, 12, //
    13, 14, 15, 16, //
  };
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  std::vector<float> bias_data{1, 2, 3, 4};
  std::vector<float> ref_output_data{
    71,  0, 99,  0,  //
    167, 0, 227, 28, //
  };

  Tensor input_tensor =
    makeInputTensor<DataType::S16>(input_shape, 0.25, 0, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::S16>(filter_shape, 0.2, 0, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::S64>(bias_shape, 0.25 * 0.2, 0, bias_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.5, 0);
  Tensor scratchpad(DataType::S64, Shape({}), {}, "");

  DepthwiseConv2DParams params{};
  params.padding = Padding::VALID;
  params.depth_multiplier = 2;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  DepthwiseConv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &scratchpad,
                         params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(scratchpad);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST_F(DepthwiseConv2DTest, SInt16_CWQ_weights)
{
  const int output_channels = 4;
  Shape input_shape{1, 4, 2, 2};
  Shape filter_shape{1, 2, 2, output_channels};
  Shape bias_shape{4};
  std::vector<int32_t> ref_output_shape{1, 2, 1, output_channels};

  std::vector<float> input_data{
    1,  2,  7,  8,  //
    3,  4,  9,  10, //
    5,  6,  11, 12, //
    13, 14, 15, 16, //
  };
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  std::vector<float> bias_data{1, 2, 3, 4};
  std::vector<float> ref_output_data{
    71,  0, 99,  0,  //
    167, 0, 227, 28, //
  };

  float input_scale = 0.25;
  std::vector<float> filter_scales{0.2f, 1.f, 0.5f, 0.1f};
  std::vector<float> bias_scales;
  for (int i = 0; i < output_channels; ++i)
    bias_scales.push_back(filter_scales[i] * input_scale);
  std::vector<int32_t> zerop(4, 0);
  Tensor input_tensor =
    makeInputTensor<DataType::S16>(input_shape, input_scale, 0, input_data, _memory_manager.get());
  Tensor filter_tensor = makeInputTensor<DataType::S16>(filter_shape, filter_scales, zerop, 3,
                                                        filter_data, _memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S64>(bias_shape, bias_scales, zerop, 0, bias_data,
                                                      _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.5, 0);
  Tensor scratchpad(DataType::S16, Shape({}), {}, "");

  DepthwiseConv2DParams params{};
  params.padding = Padding::VALID;
  params.depth_multiplier = 2;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  DepthwiseConv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &scratchpad,
                         params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(scratchpad);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST_F(DepthwiseConv2DTest, Uint8_CWQ_weights)
{
  const int output_channels = 4;
  Shape input_shape{1, 3, 2, 2};
  Shape filter_shape{1, 2, 2, output_channels};
  Shape bias_shape{4};
  std::vector<int32_t> ref_output_shape{1, 2, 1, output_channels};

  std::vector<float> input_data{
    1, 2, 7,  8,  //
    3, 4, 9,  10, //
    5, 6, 11, 12, //
  };
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  std::vector<float> bias_data{1, 2, 3, 4};
  std::vector<float> ref_output_data{
    71, -34, 99,  -20, //
    91, -26, 127, -4,  //
  };

  std::pair<float, int32_t> input_quant_param = quantizationParams<uint8_t>(0, 16);
  std::pair<float, int32_t> output_quant_param = quantizationParams<uint8_t>(-127, 128);

  std::vector<std::pair<float, int32_t>> filter_quant_params;
  filter_quant_params.push_back(quantizationParams<uint8_t>(-9, 13));
  filter_quant_params.push_back(quantizationParams<uint8_t>(-14, 10));
  filter_quant_params.push_back(quantizationParams<uint8_t>(-11, 15));
  filter_quant_params.push_back(quantizationParams<uint8_t>(-16, 12));

  std::vector<float> filter_scales;
  std::vector<int32_t> filter_zerops;
  for (auto iter : filter_quant_params)
  {
    filter_scales.push_back(iter.first);
    filter_zerops.push_back(iter.second);
  }

  std::vector<float> bias_scales;
  for (int i = 0; i < output_channels; ++i)
    bias_scales.push_back(filter_quant_params[i].first * input_quant_param.first);
  std::vector<int32_t> zerop(output_channels, 0);

  Tensor input_tensor =
    makeInputTensor<DataType::U8>(input_shape, input_quant_param.first, input_quant_param.second,
                                  input_data, _memory_manager.get());
  Tensor filter_tensor = makeInputTensor<DataType::U8>(filter_shape, filter_scales, filter_zerops,
                                                       3, filter_data, _memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S32>(bias_shape, bias_scales, zerop, 0, bias_data,
                                                      _memory_manager.get());
  Tensor output_tensor =
    makeOutputTensor(DataType::U8, output_quant_param.first, output_quant_param.second);
  Tensor scratchpad(DataType::U8, Shape({}), {}, "");

  DepthwiseConv2DParams params{};
  params.padding = Padding::VALID;
  params.depth_multiplier = 2;
  params.stride_height = 1;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::NONE;

  DepthwiseConv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &scratchpad,
                         params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(scratchpad);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(ref_output_data, output_quant_param.first));
}

TEST_F(DepthwiseConv2DTest, SInt8_CWQ_weights)
{
  const int output_channels = 4;
  Shape input_shape{1, 3, 2, 2};
  Shape filter_shape{1, 2, 2, output_channels};
  Shape bias_shape{4};
  std::vector<int32_t> ref_output_shape{1, 2, 1, output_channels};

  std::vector<float> input_data{
    1, 2, 7,  8,  //
    3, 4, 9,  10, //
    5, 6, 11, 12, //
  };
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  std::vector<float> bias_data{1, 2, 3, 4};
  std::vector<float> ref_output_data{
    71, -34, 99,  -20, //
    91, -26, 127, -4,  //
  };

  std::pair<float, int32_t> input_quant_param = quantizationParams<int8_t>(-128, 127);
  std::pair<float, int32_t> output_quant_param = quantizationParams<int8_t>(-127, 128);

  std::vector<std::pair<float, int32_t>> filter_quant_params;
  filter_quant_params.push_back(std::pair<float, int32_t>(0.5, 0));
  filter_quant_params.push_back(std::pair<float, int32_t>(0.25, 0));
  filter_quant_params.push_back(std::pair<float, int32_t>(1, 0));
  filter_quant_params.push_back(std::pair<float, int32_t>(0.125, 0));

  std::vector<float> filter_scales;
  std::vector<int32_t> filter_zerops;
  for (auto iter : filter_quant_params)
  {
    filter_scales.push_back(iter.first);
    filter_zerops.push_back(iter.second);
  }

  std::vector<float> bias_scales;
  for (int i = 0; i < output_channels; ++i)
    bias_scales.push_back(filter_quant_params[i].first * input_quant_param.first);
  std::vector<int32_t> zerop(output_channels, 0);

  Tensor input_tensor =
    makeInputTensor<DataType::S8>(input_shape, input_quant_param.first, input_quant_param.second,
                                  input_data, _memory_manager.get());
  Tensor filter_tensor = makeInputTensor<DataType::S8>(filter_shape, filter_scales, filter_zerops,
                                                       3, filter_data, _memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S32>(bias_shape, bias_scales, zerop, 0, bias_data,
                                                      _memory_manager.get());
  Tensor output_tensor =
    makeOutputTensor(DataType::S8, output_quant_param.first, output_quant_param.second);
  Tensor scratchpad(DataType::S8, Shape({}), {}, "");

  DepthwiseConv2DParams params{};
  params.padding = Padding::VALID;
  params.depth_multiplier = 2;
  params.stride_height = 1;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::NONE;

  DepthwiseConv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &scratchpad,
                         params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(scratchpad);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(ref_output_data, output_quant_param.first));
}

TEST_F(DepthwiseConv2DTest, InvalidBiasType_NEG)
{
  Shape input_shape{1, 4, 2, 2};
  Shape filter_shape{1, 2, 2, 4};
  Shape bias_shape{4};
  std::vector<float> input_data{
    1,  2,  7,  8,  //
    3,  4,  9,  10, //
    5,  6,  11, 12, //
    13, 14, 15, 16, //
  };
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  std::vector<int32_t> bias_data{1, 2, 3, 4};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S32>(bias_shape, bias_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor scratchpad(DataType::FLOAT32, Shape({}), {}, "");

  DepthwiseConv2DParams params{};
  params.padding = Padding::VALID;
  params.depth_multiplier = 2;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  DepthwiseConv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &scratchpad,
                         params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(DepthwiseConv2DTest, InOutTypeMismatch_NEG)
{
  Shape input_shape{1, 4, 2, 2};
  Shape filter_shape{1, 2, 2, 4};
  Shape bias_shape{4};
  std::vector<float> input_data{
    1,  2,  7,  8,  //
    3,  4,  9,  10, //
    5,  6,  11, 12, //
    13, 14, 15, 16, //
  };
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  std::vector<float> bias_data{1, 2, 3, 4};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8);
  Tensor scratchpad(DataType::U8, Shape({}), {}, "");

  DepthwiseConv2DParams params{};
  params.padding = Padding::VALID;
  params.depth_multiplier = 2;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  DepthwiseConv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &scratchpad,
                         params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(DepthwiseConv2DTest, InvalidInputShape_NEG)
{
  Shape input_shape{4, 2, 2};
  Shape filter_shape{2, 2, 4};
  Shape bias_shape{4};
  std::vector<float> input_data{
    1,  2,  7,  8,  //
    3,  4,  9,  10, //
    5,  6,  11, 12, //
    13, 14, 15, 16, //
  };
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  std::vector<float> bias_data{1, 2, 3, 4};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor scratchpad(DataType::FLOAT32, Shape({}), {}, "");

  DepthwiseConv2DParams params{};
  params.padding = Padding::VALID;
  params.depth_multiplier = 2;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  DepthwiseConv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &scratchpad,
                         params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(DepthwiseConv2DTest, InvalidFilterShape_NEG)
{
  Shape input_shape{1, 4, 2, 2};
  Shape filter_shape{2, 1, 2, 4};
  Shape bias_shape{4};
  std::vector<float> input_data{
    1,  2,  7,  8,  //
    3,  4,  9,  10, //
    5,  6,  11, 12, //
    13, 14, 15, 16, //
  };
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  std::vector<float> bias_data{1, 2, 3, 4};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor scratchpad(DataType::FLOAT32, Shape({}), {}, "");

  DepthwiseConv2DParams params{};
  params.padding = Padding::VALID;
  params.depth_multiplier = 2;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  DepthwiseConv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &scratchpad,
                         params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(DepthwiseConv2DTest, InvalidBiasDim_NEG)
{
  Shape input_shape{1, 4, 2, 2};
  Shape filter_shape{1, 2, 4, 2};
  Shape bias_shape{4};
  std::vector<float> input_data{
    1,  2,  7,  8,  //
    3,  4,  9,  10, //
    5,  6,  11, 12, //
    13, 14, 15, 16, //
  };
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  std::vector<float> bias_data{1, 2, 3, 4};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor scratchpad(DataType::FLOAT32, Shape({}), {}, "");

  DepthwiseConv2DParams params{};
  params.padding = Padding::VALID;
  params.depth_multiplier = 2;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  DepthwiseConv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &scratchpad,
                         params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
