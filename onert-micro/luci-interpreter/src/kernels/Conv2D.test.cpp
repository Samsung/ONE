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

// TODO enable it
#if 0
#include "kernels/Conv2D.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class Conv2DTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(Conv2DTest, Float)
{
  Shape input_shape{1, 4, 3, 2};
  Shape filter_shape{2, 2, 2, 2};
  Shape bias_shape{2};
  std::vector<float> input_data{
    1,  2,  3,  4,  5,  6,  // row = 0
    7,  8,  9,  10, 11, 12, // row = 1
    13, 14, 15, 16, 17, 18, // row = 2
    19, 20, 21, 22, 23, 24, // row = 3
  };
  std::vector<float> filter_data{
    1,  2,  -3, -4, // out = 0, row = 0
    -5, 6,  -7, 8,  // out = 1, row = 0
    4,  -2, 3,  -1, // out = 0, row = 1
    -8, -6, 7,  5,  // out = 1, row = 1
  };
  std::vector<float> bias_data{1, 2};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor im2col(DataType::FLOAT32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  kernel.configure();
  _memory_manager->allocate_memory(im2col);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{
    11, 16, 7, 20, // row = 0
    0,  40, 0, 44, // row = 1
  };
  std::vector<int32_t> ref_output_shape{1, 2, 2, 2};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(Conv2DTest, FloatPointwise)
{
  Shape input_shape{1, 2, 2, 2};
  Shape filter_shape{2, 1, 1, 2};
  Shape bias_shape{2};
  std::vector<float> input_data{
    1, 2, // row = 0, col = 0
    3, 4, // row = 0, col = 1
    5, 6, // row = 1, col = 0
    7, 8, // row = 1, col = 1
  };
  std::vector<float> filter_data{
    -1, 2, // out = 0
    -3, 4, // out = 1
  };
  std::vector<float> bias_data{1, 2};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor im2col(DataType::FLOAT32, Shape({}), {}, "");

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 1;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  kernel.configure();
  _memory_manager->allocate_memory(im2col);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{
    4, 7,  6,  9,  // row = 0
    8, 11, 10, 13, // row = 1
  };
  std::vector<int32_t> ref_output_shape{1, 2, 2, 2};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(Conv2DTest, FloatCheck)
{
  Shape input_shape{2, 2, 4, 1};
  Shape filter_shape{3, 2, 2, 1};
  Shape bias_shape{3};
  std::vector<float> input_data{
    // First batch
    1, 1, 1, 1, // row = 1
    2, 2, 2, 2, // row = 2
    // Second batch
    1, 2, 3, 4, // row = 1
    1, 2, 3, 4, // row = 2
  };
  std::vector<float> filter_data{
    1,  2,  3,  4, // first 2x2 filter
    -1, 1,  -1, 1, // second 2x2 filter
    -1, -1, 1,  1, // third 2x2 filter
  };
  std::vector<float> bias_data{1, 2, 3};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor im2col(DataType::FLOAT32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 2;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::NONE;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(im2col);
  kernel.execute();

  std::vector<float> ref_output_data{
    18, 2, 5, // first batch, left
    18, 2, 5, // first batch, right
    17, 4, 3, // second batch, left
    37, 4, 3, // second batch, right
  };
  std::vector<int32_t> ref_output_shape{2, 1, 2, 3};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(Conv2DTest, Uint8)
{
  std::vector<float> input_data{
    // First batch
    1, 1, 1, 1, // row = 1
    2, 2, 2, 2, // row = 2
                // Second batch
    1, 2, 3, 4, // row = 1
    1, 2, 3, 4, // row = 2
  };
  std::vector<float> filter_data{
    1,  2,  3,  4, // first 2x2 filter
    -1, 1,  -1, 1, // second 2x2 filter
    -1, -1, 1,  1, // third 2x2 filter
  };
  std::vector<float> bias_data{1, 2, 3};

  std::pair<float, int32_t> input_quant_param = quantizationParams<uint8_t>(-63.5, 64);
  std::pair<float, int32_t> output_quant_param = quantizationParams<uint8_t>(-127, 128);

  Tensor input_tensor =
    makeInputTensor<DataType::U8>({2, 2, 4, 1}, input_quant_param.first, input_quant_param.second,
                                  input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::U8>({3, 2, 2, 1}, input_quant_param.first, input_quant_param.second,
                                  filter_data, _memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S32>(
    {3}, input_quant_param.first * input_quant_param.first, 0, bias_data, _memory_manager.get());
  Tensor im2col(DataType::U8, Shape({}), {}, "");
  Tensor output_tensor =
    makeOutputTensor(DataType::U8, output_quant_param.first, output_quant_param.second);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 2;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::NONE;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(im2col);
  kernel.execute();

  std::vector<float> ref_output_data{
    18, 2, 5, // first batch, left
    18, 2, 5, // first batch, right
    17, 4, 3, // second batch, left
    37, 4, 3, // second batch, right
  };
  std::vector<int32_t> ref_output_shape{2, 1, 2, 3};
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(Conv2DTest, Uint8_CWQ)
{
  const int output_channels = 3;
  std::vector<float> input_data{
    // First batch
    1, 1, 1, 1, // row = 1
    2, 2, 2, 2, // row = 2
                // Second batch
    1, 2, 3, 4, // row = 1
    1, 2, 3, 4, // row = 2
  };
  std::vector<float> filter_data{
    1,  2,  3,  4, // first 2x2 filter
    -1, 1,  -1, 1, // second 2x2 filter
    -1, -1, 1,  1, // third 2x2 filter
  };
  std::vector<float> bias_data{1, 2, 3};
  Shape filter_shape{output_channels, 2, 2, 1};

  std::pair<float, int32_t> input_quant_param = quantizationParams<uint8_t>(0, 4);
  std::pair<float, int32_t> output_quant_param = quantizationParams<uint8_t>(-127, 128);

  std::vector<std::pair<float, int32_t>> filter_quant_params;
  filter_quant_params.push_back(quantizationParams<uint8_t>(0, 4));
  filter_quant_params.push_back(quantizationParams<uint8_t>(-1, 1));
  filter_quant_params.push_back(quantizationParams<uint8_t>(-1, 1));

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
    makeInputTensor<DataType::U8>({2, 2, 4, 1}, input_quant_param.first, input_quant_param.second,
                                  input_data, _memory_manager.get());
  Tensor filter_tensor = makeInputTensor<DataType::U8>(filter_shape, filter_scales, filter_zerops,
                                                       0, filter_data, _memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S32>({output_channels}, bias_scales, zerop, 0,
                                                      bias_data, _memory_manager.get());
  Tensor im2col(DataType::U8, Shape({}), {}, "");
  Tensor output_tensor =
    makeOutputTensor(DataType::U8, output_quant_param.first, output_quant_param.second);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 2;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::NONE;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(im2col);
  kernel.execute();

  std::vector<float> ref_output_data{
    18, 2, 5, // first batch, left
    18, 2, 5, // first batch, right
    17, 4, 3, // second batch, left
    37, 4, 3, // second batch, right
  };
  std::vector<int32_t> ref_output_shape{2, 1, 2, 3};
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(Conv2DTest, SInt8_CWQ)
{
  const int output_channels = 3;
  std::vector<float> input_data{
    // First batch
    1, 1, 1, 1, // row = 1
    2, 2, 2, 2, // row = 2
                // Second batch
    1, 2, 3, 4, // row = 1
    1, 2, 3, 4, // row = 2
  };
  std::vector<float> filter_data{
    1,  2,  3,  4, // first 2x2 filter
    -1, 1,  -1, 1, // second 2x2 filter
    -1, -1, 1,  1, // third 2x2 filter
  };
  std::vector<float> bias_data{1, 2, 3};
  Shape filter_shape{output_channels, 2, 2, 1};

  std::pair<float, int32_t> input_quant_param = quantizationParams<int8_t>(0, 4);
  std::pair<float, int32_t> output_quant_param = quantizationParams<int8_t>(-127, 128);

  std::vector<std::pair<float, int32_t>> filter_quant_params;
  filter_quant_params.push_back(std::pair<float, int32_t>(0.5, 0));
  filter_quant_params.push_back(std::pair<float, int32_t>(0.25, 0));
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
    makeInputTensor<DataType::S8>({2, 2, 4, 1}, input_quant_param.first, input_quant_param.second,
                                  input_data, _memory_manager.get());
  Tensor filter_tensor = makeInputTensor<DataType::S8>(filter_shape, filter_scales, filter_zerops,
                                                       0, filter_data, _memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S32>({output_channels}, bias_scales, zerop, 0,
                                                      bias_data, _memory_manager.get());
  Tensor im2col(DataType::S8, Shape({}), {}, "");
  Tensor output_tensor =
    makeOutputTensor(DataType::S8, output_quant_param.first, output_quant_param.second);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 2;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::NONE;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(im2col);
  kernel.execute();

  std::vector<float> ref_output_data{
    18, 2, 5, // first batch, left
    18, 2, 5, // first batch, right
    17, 4, 3, // second batch, left
    37, 4, 3, // second batch, right
  };
  std::vector<int32_t> ref_output_shape{2, 1, 2, 3};
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(Conv2DTest, SInt16)
{
  Shape input_shape{1, 4, 3, 2};
  Shape filter_shape{2, 2, 2, 2};
  Shape bias_shape{2};
  std::vector<int32_t> ref_output_shape{1, 2, 2, 2};

  std::vector<float> input_data{
    1,  2,  3,  4,  5,  6,  // row = 0
    7,  8,  9,  10, 11, 12, // row = 1
    13, 14, 15, 16, 17, 18, // row = 2
    19, 20, 21, 22, 23, 24, // row = 3
  };
  std::vector<float> filter_data{
    1,  2,  -3, -4, // out = 0, row = 0
    -5, 6,  -7, 8,  // out = 1, row = 0
    4,  -2, 3,  -1, // out = 0, row = 1
    -8, -6, 7,  5,  // out = 1, row = 1
  };
  std::vector<float> bias_data{1, 2};
  std::vector<float> ref_output_data{
    11, 16, 7, 20, // row = 0
    0,  40, 0, 44, // row = 1
  };

  Tensor input_tensor =
    makeInputTensor<DataType::S16>(input_shape, 0.25, 0, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::S16>(filter_shape, 0.2, 0, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::S64>(bias_shape, 0.25 * 0.2, 0, bias_data, _memory_manager.get());
  Tensor im2col(DataType::S16, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.5, 0);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(im2col);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST_F(Conv2DTest, SInt16_CWQ_weights)
{
  Shape input_shape{1, 2, 2, 2};  // Batch x H x W x C
  Shape filter_shape{3, 1, 1, 2}; // Out channels x H x W x In Channels
  Shape bias_shape{3};
  std::vector<int32_t> ref_output_shape{1, 2, 2, 3};

  std::vector<float> input_data{
    1, 2, // row = 0, col 0
    3, 4, // row = 0, col 1
    5, 6, // row = 1, col 0
    7, 8, // row = 1, col 1
  };
  std::vector<float> filter_data{
    4, -3, // out = 0
    1, -3, // out = 1
    5, -3, // out = 2
  };
  std::vector<float> bias_data{1, 10, 5};
  std::vector<float> ref_output_data{
    0, 5, 4,  // row 0, col 0
    1, 1, 8,  // row 0, col 1
    3, 0, 12, // row 1, col 0
    5, 0, 16, // row 1, col 1
  };

  float input_scale = 0.25f;
  float output_scale = 0.05f;
  std::vector<float> filter_scales = {0.25f, 0.2f, 0.1f};
  std::vector<float> bias_scales;
  for (int i = 0; i < filter_scales.size(); ++i)
    bias_scales.push_back(filter_scales[i] * input_scale);
  std::vector<int32_t> zerop = {0, 0, 0};

  Tensor input_tensor =
    makeInputTensor<DataType::S16>(input_shape, input_scale, 0, input_data, _memory_manager.get());
  Tensor filter_tensor = makeInputTensor<DataType::S16>(filter_shape, filter_scales, zerop, 0,
                                                        filter_data, _memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S64>(bias_shape, bias_scales, zerop, 0, bias_data,
                                                      _memory_manager.get());
  Tensor im2col(DataType::S16, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::S16, output_scale, 0);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 1;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(im2col);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST_F(Conv2DTest, Unsupported_Type_Configure_NEG)
{
  Shape input_shape{1, 4, 3, 2};
  Shape filter_shape{2, 2, 2, 2};
  Shape bias_shape{2};
  std::vector<int32_t> input_data{
    1,  2,  3,  4,  5,  6,  // row = 0
    7,  8,  9,  10, 11, 12, // row = 1
    13, 14, 15, 16, 17, 18, // row = 2
    19, 20, 21, 22, 23, 24, // row = 3
  };
  std::vector<float> filter_data{
    1,  2,  -3, -4, // out = 0, row = 0
    -5, 6,  -7, 8,  // out = 1, row = 0
    4,  -2, 3,  -1, // out = 0, row = 1
    -8, -6, 7,  5,  // out = 1, row = 1
  };
  std::vector<float> bias_data{1, 2};
  Tensor input_tensor =
    makeInputTensor<DataType::S32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor im2col(DataType::FLOAT32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(Conv2DTest, Invalid_Bias_Type_NEG)
{
  Shape input_shape{1, 4, 3, 2};
  Shape filter_shape{2, 2, 2, 2};
  Shape bias_shape{2};
  std::vector<float> input_data{
    1,  2,  3,  4,  5,  6,  // row = 0
    7,  8,  9,  10, 11, 12, // row = 1
    13, 14, 15, 16, 17, 18, // row = 2
    19, 20, 21, 22, 23, 24, // row = 3
  };
  std::vector<float> filter_data{
    1,  2,  -3, -4, // out = 0, row = 0
    -5, 6,  -7, 8,  // out = 1, row = 0
    4,  -2, 3,  -1, // out = 0, row = 1
    -8, -6, 7,  5,  // out = 1, row = 1
  };
  std::vector<uint8_t> bias_data{1, 2};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::U8>(bias_shape, bias_data, _memory_manager.get());
  Tensor im2col(DataType::FLOAT32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(Conv2DTest, Invalid_Bias_Data_NEG)
{
  Shape input_shape{1, 4, 3, 2};
  Shape filter_shape{2, 2, 2, 2};
  Shape bias_shape{3};
  std::vector<float> input_data{
    1,  2,  3,  4,  5,  6,  // row = 0
    7,  8,  9,  10, 11, 12, // row = 1
    13, 14, 15, 16, 17, 18, // row = 2
    19, 20, 21, 22, 23, 24, // row = 3
  };
  std::vector<float> filter_data{
    1,  2,  -3, -4, // out = 0, row = 0
    -5, 6,  -7, 8,  // out = 1, row = 0
    4,  -2, 3,  -1, // out = 0, row = 1
    -8, -6, 7,  5,  // out = 1, row = 1
  };
  std::vector<float> bias_data{1, 2, 3};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor im2col(DataType::FLOAT32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(Conv2DTest, Invalid_Input_Shape_NEG)
{
  Shape input_shape{1, 4, 6, 1};
  Shape filter_shape{2, 2, 2, 2};
  Shape bias_shape{2};
  std::vector<float> input_data{
    1,  2,  3,  4,  5,  6,  // row = 0
    7,  8,  9,  10, 11, 12, // row = 1
    13, 14, 15, 16, 17, 18, // row = 2
    19, 20, 21, 22, 23, 24, // row = 3
  };
  std::vector<float> filter_data{
    1,  2,  -3, -4, // out = 0, row = 0
    -5, 6,  -7, 8,  // out = 1, row = 0
    4,  -2, 3,  -1, // out = 0, row = 1
    -8, -6, 7,  5,  // out = 1, row = 1
  };
  std::vector<float> bias_data{1, 2};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor im2col(DataType::FLOAT32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(Conv2DTest, Invalid_fused_act_tanh_NEG)
{
  Shape input_shape{1, 4, 3, 2};
  Shape filter_shape{2, 2, 2, 2};
  Shape bias_shape{2};
  std::vector<float> input_data{
    1,  2,  3,  4,  5,  6,  // row = 0
    7,  8,  9,  10, 11, 12, // row = 1
    13, 14, 15, 16, 17, 18, // row = 2
    19, 20, 21, 22, 23, 24, // row = 3
  };
  std::vector<float> filter_data{
    1,  2,  -3, -4, // out = 0, row = 0
    -5, 6,  -7, 8,  // out = 1, row = 0
    4,  -2, 3,  -1, // out = 0, row = 1
    -8, -6, 7,  5,  // out = 1, row = 1
  };
  std::vector<float> bias_data{1, 2};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data, _memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data, _memory_manager.get());
  Tensor im2col(DataType::FLOAT32, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::TANH;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, &im2col, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
#endif
