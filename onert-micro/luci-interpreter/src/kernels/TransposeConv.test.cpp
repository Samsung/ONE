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

#include "kernels/TransposeConv.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T, typename B>
void Check(std::initializer_list<int32_t> output_shape_shape,
           std::initializer_list<int32_t> weight_shape, std::initializer_list<int32_t> input_shape,
           std::initializer_list<int32_t> bias_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<int32_t> output_shape_data, std::initializer_list<T> weight_data,
           std::initializer_list<T> input_data, std::initializer_list<B> bias_data,
           std::initializer_list<T> output_data, luci::Padding padding, int32_t stride_height,
           int32_t stride_width)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  constexpr DataType element_type = getElementType<T>();
  Tensor output_shape_tensor =
    makeInputTensor<DataType::S32>(output_shape_shape, output_shape_data, memory_manager.get());
  Tensor weight_tensor =
    makeInputTensor<element_type>(weight_shape, weight_data, memory_manager.get());
  Tensor input_data_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());

  DataType scratch_data_type = element_type == DataType::S16 ? DataType::S64 : DataType::S32;
  Tensor scratch_tensor(scratch_data_type, Shape({}), {}, "");
  Tensor output_tensor = makeOutputTensor(element_type);

  TransposeConvParams params{};
  params.padding = padding;
  params.stride_height = stride_height;
  params.stride_width = stride_width;

  if (bias_data.size() != 0)
  {
    Tensor bias_tensor =
      makeInputTensor<getElementType<B>()>(bias_shape, bias_data, memory_manager.get());
    TransposeConv kernel(&output_shape_tensor, &weight_tensor, &input_data_tensor, &bias_tensor,
                         &output_tensor, &scratch_tensor, params);
    kernel.configure();
    memory_manager->allocate_memory(output_tensor);
    memory_manager->allocate_memory(scratch_tensor);
    kernel.execute();
  }
  else
  {
    TransposeConv kernel(&output_shape_tensor, &weight_tensor, &input_data_tensor, nullptr,
                         &output_tensor, &scratch_tensor, params);
    kernel.configure();
    memory_manager->allocate_memory(output_tensor);
    memory_manager->allocate_memory(scratch_tensor);
    kernel.execute();
  }
  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
}

TEST(TransposeConvTest, FloatSimple)
{
  Check<float, float>(
    /*output_shape_shape=*/{4}, /*weight_shape=*/{1, 3, 3, 1}, /*input_shape=*/{1, 4, 4, 1},
    /*bias_shape=*/{}, /*output_shape=*/{1, 4, 4, 1}, /*output_shape_data=*/{1, 4, 4, 1},
    /*weight_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9},
    /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    /*bias_data=*/{},
    /*output_data=*/{29, 62, 83, 75, 99, 192, 237, 198, 207, 372, 417, 330, 263, 446, 485, 365},
    /*params.padding=*/luci::Padding::SAME, /*stride_height=*/1, /*stride_width=*/1);

  SUCCEED();
}

TEST(TransposeConvTest, FloatTwoFiltersTest)
{
  Check<float, float>(
    /*output_shape_shape=*/{4}, /*weight_shape=*/{1, 3, 3, 2}, /*input_shape=*/{1, 4, 4, 2},
    /*bias_shape=*/{}, /*output_shape=*/{1, 4, 4, 1}, /*output_shape_data=*/{1, 4, 4, 1},
    /*weight_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
    /*input_data=*/{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32},
    /*bias_data=*/{},
    /*output_data=*/
    {184, 412, 568, 528, 678, 1347, 1689, 1434, 1494, 2715, 3057, 2442, 1968, 3352, 3652, 2760},
    /*params.padding=*/luci::Padding::SAME, /*stride_height=*/1, /*stride_width=*/1);

  SUCCEED();
}

TEST(TransposeConvTest, SimpleBiasTest)
{
  Check<float, float>(
    /*output_shape_shape=*/{4}, /*weight_shape=*/{2, 3, 3, 1},
    /*input_shape=*/{1, 2, 2, 1},
    /*bias_shape=*/{2}, /*output_shape=*/{1, 4, 4, 1}, /*output_shape_data=*/{1, 5, 5, 2},
    /*weight_data=*/{1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18},
    /*input_data=*/{1, 2, 3, 4},
    /*bias_data=*/{3, 4},
    /*output_data=*/{4,  6,  6,  8,  10, 14, 9,  12, 13, 16, 10,  12,  12, 14, 28, 32, 21,
                     24, 25, 28, 19, 24, 27, 32, 65, 76, 45, 52,  57,  64, 24, 28, 30, 34,
                     64, 72, 39, 44, 47, 52, 42, 46, 48, 52, 106, 114, 63, 68, 71, 76},
    /*params.padding=*/luci::Padding::VALID, /*stride_height=*/2, /*stride_width=*/2);

  SUCCEED();
}

TEST(TransposeConvTest, UInt8)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  std::vector<float> input_data{1, 2, 3, 4};
  std::vector<float> filter_data{1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  std::vector<float> bias_data{3, 4};
  std::vector<int32_t> output_shape_data{1, 5, 5, 2};
  std::vector<float> ref_output_data{
    4,  6,  6,  8,  10,  14,  9,  12, 13, 16, //
    10, 12, 12, 14, 28,  32,  21, 24, 25, 28, //
    19, 24, 27, 32, 65,  76,  45, 52, 57, 64, //
    24, 28, 30, 34, 64,  72,  39, 44, 47, 52, //
    42, 46, 48, 52, 106, 114, 63, 68, 71, 76, //
  };

  // Choose quantization parameters carefully.
  auto input_quant = quantizationParams<uint8_t>(-8.0, 7.9375);  // s = 1 / 16, zp = 128
  auto filter_quant = quantizationParams<uint8_t>(-24.0, 39.75); // s = 1 / 4, zp = 96
  auto output_quant = quantizationParams<uint8_t>(-64.0, 191.0); // s = 1, zp = 64

  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 2, 1}, input_quant.first, input_quant.second, input_data, memory_manager.get());
  Tensor filter_tensor = makeInputTensor<DataType::U8>(
    {2, 3, 3, 1}, filter_quant.first, filter_quant.second, filter_data, memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S32>({2}, input_quant.first * filter_quant.first,
                                                      0, bias_data, memory_manager.get());
  Tensor output_shape_tensor =
    makeInputTensor<DataType::S32>({4}, output_shape_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, output_quant.first, output_quant.second);

  DataType scratch_data_type =
    input_tensor.element_type() == DataType::S16 ? DataType::S64 : DataType::S32;
  Tensor scratch_tensor(scratch_data_type, Shape({}), {}, "");

  TransposeConvParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 2;

  TransposeConv kernel(&output_shape_tensor, &filter_tensor, &input_tensor, &bias_tensor,
                       &output_tensor, &scratch_tensor, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  memory_manager->allocate_memory(scratch_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape_data));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST(TransposeConvTest, UInt8_CWQ)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  const int32_t output_channels = 2;
  std::vector<float> input_data{1, 2, 3, 4};
  std::vector<float> filter_data{1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  std::vector<float> bias_data{3, 4};
  std::vector<int32_t> output_shape_data{1, 5, 5, 2};
  std::vector<float> ref_output_data{
    4,  6,  6,  8,  10,  14,  9,  12, 13, 16, //
    10, 12, 12, 14, 28,  32,  21, 24, 25, 28, //
    19, 24, 27, 32, 65,  76,  45, 52, 57, 64, //
    24, 28, 30, 34, 64,  72,  39, 44, 47, 52, //
    42, 46, 48, 52, 106, 114, 63, 68, 71, 76, //
  };

  // Choose quantization parameters carefully.
  auto input_quant = quantizationParams<uint8_t>(-8.0, 7.9375);  // s = 1 / 16, zp = 128
  auto output_quant = quantizationParams<uint8_t>(-64.0, 191.0); // s = 1, zp = 64

  std::vector<std::pair<float, int32_t>> filter_quant_params;
  filter_quant_params.push_back(quantizationParams<uint8_t>(0, 17));
  filter_quant_params.push_back(quantizationParams<uint8_t>(0, 18));

  std::vector<float> filter_scales;
  std::vector<int32_t> filter_zerops;
  for (auto iter : filter_quant_params)
  {
    filter_scales.push_back(iter.first);
    filter_zerops.push_back(iter.second);
  }

  std::vector<float> bias_scales;
  for (int i = 0; i < output_channels; ++i)
    bias_scales.push_back(filter_quant_params[i].first * input_quant.first);
  std::vector<int32_t> zerop(output_channels, 0);

  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 2, 1}, input_quant.first, input_quant.second, input_data, memory_manager.get());
  Tensor filter_tensor = makeInputTensor<DataType::U8>(
    {output_channels, 3, 3, 1}, filter_scales, filter_zerops, 0, filter_data, memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S32>({output_channels}, bias_scales, zerop, 0,
                                                      bias_data, memory_manager.get());
  Tensor output_shape_tensor =
    makeInputTensor<DataType::S32>({4}, output_shape_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, output_quant.first, output_quant.second);

  DataType scratch_data_type =
    input_tensor.element_type() == DataType::S16 ? DataType::S64 : DataType::S32;
  Tensor scratch_tensor(scratch_data_type, Shape({}), {}, "");

  TransposeConvParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 2;

  TransposeConv kernel(&output_shape_tensor, &filter_tensor, &input_tensor, &bias_tensor,
                       &output_tensor, &scratch_tensor, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  memory_manager->allocate_memory(scratch_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape_data));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST(TransposeConvTest, SInt16)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  std::vector<float> input_data{1, 2, 3, 4};
  std::vector<float> filter_data{1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  std::vector<float> bias_data{3, 4};
  std::vector<int32_t> output_shape_data{1, 5, 5, 2};
  std::vector<float> ref_output_data{
    4,  6,  6,  8,  10,  14,  9,  12, 13, 16, //
    10, 12, 12, 14, 28,  32,  21, 24, 25, 28, //
    19, 24, 27, 32, 65,  76,  45, 52, 57, 64, //
    24, 28, 30, 34, 64,  72,  39, 44, 47, 52, //
    42, 46, 48, 52, 106, 114, 63, 68, 71, 76, //
  };

  Tensor input_tensor =
    makeInputTensor<DataType::S16>({1, 2, 2, 1}, 0.25, 0, input_data, memory_manager.get());
  Tensor filter_tensor =
    makeInputTensor<DataType::S16>({2, 3, 3, 1}, 0.2, 0, filter_data, memory_manager.get());
  Tensor bias_tensor =
    makeInputTensor<DataType::S64>({2}, 0.25 * 0.2, 0, bias_data, memory_manager.get());
  Tensor output_shape_tensor =
    makeInputTensor<DataType::S32>({4}, output_shape_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.5, 0);

  DataType scratch_data_type =
    input_tensor.element_type() == DataType::S16 ? DataType::S64 : DataType::S32;
  Tensor scratch_tensor(scratch_data_type, Shape({}), {}, "");

  TransposeConvParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 2;

  TransposeConv kernel(&output_shape_tensor, &filter_tensor, &input_tensor, &bias_tensor,
                       &output_tensor, &scratch_tensor, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  memory_manager->allocate_memory(scratch_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape_data));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST(TransposeConvTest, SInt16_CWQ_weights)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  const int output_channels = 2;
  const Shape input_shape{1, 2, 2, 1};
  const Shape filter_shape{output_channels, 3, 3, 1};
  const Shape bias_shape{output_channels};
  std::vector<int32_t> output_shape_data{1, 5, 5, output_channels};

  std::vector<float> input_data{1, 2, 3, 4};
  std::vector<float> filter_data{1, 3, 5, 7, 9, 11, 13, 15, 17, 2, 4, 6, 8, 10, 12, 14, 16, 18};
  std::vector<float> bias_data{3, 4};

  std::vector<float> ref_output_data{
    4,  6,  6,  8,  10,  14,  9,  12, 13, 16, //
    10, 12, 12, 14, 28,  32,  21, 24, 25, 28, //
    19, 24, 27, 32, 65,  76,  45, 52, 57, 64, //
    24, 28, 30, 34, 64,  72,  39, 44, 47, 52, //
    42, 46, 48, 52, 106, 114, 63, 68, 71, 76, //
  };

  const float input_scale = 0.25;
  const float output_scale = 0.5;
  const std::vector<float> filter_scales{0.2f, 0.5f};
  std::vector<float> bias_scales{filter_scales[0] * input_scale, filter_scales[1] * input_scale};
  const std::vector<int32_t> zerop(2, 0);

  Tensor input_tensor =
    makeInputTensor<DataType::S16>(input_shape, input_scale, 0, input_data, memory_manager.get());
  Tensor filter_tensor = makeInputTensor<DataType::S16>(filter_shape, filter_scales, zerop, 0,
                                                        filter_data, memory_manager.get());
  Tensor bias_tensor = makeInputTensor<DataType::S64>(bias_shape, bias_scales, zerop, 0, bias_data,
                                                      memory_manager.get());
  Tensor output_shape_tensor =
    makeInputTensor<DataType::S32>({4}, output_shape_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, output_scale, 0);

  DataType scratch_data_type =
    input_tensor.element_type() == DataType::S16 ? DataType::S64 : DataType::S32;
  Tensor scratch_tensor(scratch_data_type, Shape({}), {}, "");

  TransposeConvParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 2;

  TransposeConv kernel(&output_shape_tensor, &filter_tensor, &input_tensor, &bias_tensor,
                       &output_tensor, &scratch_tensor, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  memory_manager->allocate_memory(scratch_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape_data));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
