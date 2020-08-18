/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Conv2D.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(Conv2DTest, Float)
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
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor filter_tensor = makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data);
  Tensor bias_tensor = makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{
      11, 16, 7, 20, // row = 0
      0,  40, 0, 44, // row = 1
  };
  std::vector<int32_t> ref_output_shape{1, 2, 2, 2};
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(Conv2DTest, FloatCheck)
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
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor filter_tensor = makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data);
  Tensor bias_tensor = makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 2;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::NONE;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{
      18, 2, 5, // first batch, left
      18, 2, 5, // first batch, right
      17, 4, 3, // second batch, left
      37, 4, 3, // second batch, right
  };
  std::vector<int32_t> ref_output_shape{2, 1, 2, 3};
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(Conv2DTest, Uint8)
{
  std::pair<float, int32_t> input_quant_param = quantizationParams<uint8_t>(-63.5, 64);
  std::pair<float, int32_t> output_quant_param = quantizationParams<uint8_t>(-127, 128);
  Shape bias_shape = {3};
  Tensor input_tensor{
      DataType::U8, {2, 2, 4, 1}, {{input_quant_param.first}, {input_quant_param.second}}, ""};
  Tensor filter_tensor{
      DataType::U8, {3, 2, 2, 1}, {{input_quant_param.first}, {input_quant_param.second}}, ""};
  Tensor bias_tensor{
      DataType::S32, bias_shape, {{input_quant_param.first * input_quant_param.first}, {0}}, ""};
  Tensor output_tensor =
      makeOutputTensor(DataType::U8, output_quant_param.first, output_quant_param.second);
  std::vector<uint8_t> quantized_input = quantize<uint8_t>(
      {
          // First batch
          1, 1, 1, 1, // row = 1
          2, 2, 2, 2, // row = 2
          // Second batch
          1, 2, 3, 4, // row = 1
          1, 2, 3, 4, // row = 2
      },
      input_quant_param.first, input_quant_param.second);
  std::vector<uint8_t> quantized_filter = quantize<uint8_t>(
      {
          1, 2, 3, 4,   // first 2x2 filter
          -1, 1, -1, 1, // second 2x2 filter
          -1, -1, 1, 1, // third 2x2 filter
      },
      input_quant_param.first, input_quant_param.second);
  std::vector<int32_t> bias_data =
      quantize<int32_t>({1, 2, 3}, input_quant_param.first * input_quant_param.first, 0);
  input_tensor.writeData(quantized_input.data(), quantized_input.size() * sizeof(uint8_t));
  filter_tensor.writeData(quantized_filter.data(), quantized_filter.size() * sizeof(uint8_t));
  bias_tensor.writeData(bias_data.data(), bias_data.size() * sizeof(int32_t));

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 2;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::NONE;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{
      18, 2, 5, // first batch, left
      18, 2, 5, // first batch, right
      17, 4, 3, // second batch, left
      37, 4, 3, // second batch, right
  };
  std::vector<int32_t> ref_output_shape{2, 1, 2, 3};
  EXPECT_THAT(dequantize<uint8_t>(extractTensorData<uint8_t>(output_tensor),
                                  output_quant_param.first, output_quant_param.second),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(Conv2DTest, Unsupported_Type_Configure_NEG)
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
  Tensor input_tensor = makeInputTensor<DataType::S32>(input_shape, input_data);
  Tensor filter_tensor = makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data);
  Tensor bias_tensor = makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(Conv2DTest, Invalid_Bias_Type_NEG)
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
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor filter_tensor = makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data);
  Tensor bias_tensor = makeInputTensor<DataType::U8>(bias_shape, bias_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(Conv2DTest, Invalid_Bias_Data_NEG)
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
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor filter_tensor = makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data);
  Tensor bias_tensor = makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(Conv2DTest, Invalid_Input_Shape_NEG)
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
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor filter_tensor = makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data);
  Tensor bias_tensor = makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(Conv2DTest, Unsupported_Type_Execute_NEG)
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
  std::vector<int32_t> filter_data{
      1,  2,  -3, -4, // out = 0, row = 0
      -5, 6,  -7, 8,  // out = 1, row = 0
      4,  -2, 3,  -1, // out = 0, row = 1
      -8, -6, 7,  5,  // out = 1, row = 1
  };
  std::vector<float> bias_data{1, 2};
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor filter_tensor = makeInputTensor<DataType::S32>(filter_shape, filter_data);
  Tensor bias_tensor = makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
