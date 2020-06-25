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

// for quantized Add, the error shouldn't exceed step
float GetTolerance(float min, float max)
{
  float kQuantizedStep = (max - min) / 255.0;
  return kQuantizedStep;
}

// Returns the corresponding DataType given the type T.
template <typename T> constexpr DataType getBiasType()
{
  if (std::is_same<T, float>::value)
    return DataType::FLOAT32;
  return DataType::S32;
}

float getMinValue(std::initializer_list<float> data)
{
  return 0 > std::min<float>(data) ? std::min<float>(data) : 0;
}

float getMaxValue(std::initializer_list<float> data)
{
  return 0 < std::max<float>(data) ? std::max<float>(data) : 0;
}

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> filter_shape,
           std::initializer_list<int32_t> bias_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<float> input_data, std::initializer_list<float> filter_data,
           std::initializer_list<float> bias_data, std::initializer_list<float> output_data)
{
  float kQuantizedTolerance = GetTolerance(-127, 128);
  std::pair<float, int32_t> input_quant_param = quantizationParams<T>(-63.5, 64);
  std::pair<float, int32_t> filter_quant_param = quantizationParams<T>(-63.5, 64);
  std::pair<float, int32_t> output_quant_param = quantizationParams<T>(-127, 128);

  Tensor input_tensor{getElementType<T>(),
                      input_shape,
                      {{input_quant_param.first}, {input_quant_param.second}},
                      ""};
  Tensor filter_tensor{getElementType<T>(),
                       filter_shape,
                       {{filter_quant_param.first}, {filter_quant_param.second}},
                       ""};
  Tensor bias_tensor{getBiasType<T>(), bias_shape, {}, ""};
  Tensor output_tensor =
      makeOutputTensor(getElementType<T>(), output_quant_param.first, output_quant_param.second);
  if (std::is_floating_point<T>::value)
  {
    input_tensor.writeData(input_data.begin(), input_data.size() * sizeof(T));
    filter_tensor.writeData(filter_data.begin(), filter_data.size() * sizeof(T));
    bias_tensor.writeData(bias_data.begin(), bias_data.size() * sizeof(float));
  }
  else
  {
    std::vector<T> quantized_input_value =
        quantize<T>(input_data, input_quant_param.first, input_quant_param.second);
    input_tensor.writeData(quantized_input_value.data(), quantized_input_value.size() * sizeof(T));
    std::vector<T> quantized_filter_value =
        quantize<T>(filter_data, filter_quant_param.first, filter_quant_param.second);
    filter_tensor.writeData(quantized_filter_value.data(),
                            quantized_filter_value.size() * sizeof(T));
    std::vector<int32_t> converted_bias_data;
    const float *p = bias_data.begin();
    for (int i = 0; i < bias_data.size(); i++)
    {
      converted_bias_data.push_back(static_cast<int32_t>(*p));
      p++;
    }

    bias_tensor.writeData(converted_bias_data.data(), converted_bias_data.size() * sizeof(int32_t));
  }

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

  if (std::is_floating_point<T>::value)
  {
    EXPECT_THAT(extractTensorData<T>(output_tensor),
                ElementsAreArray(ArrayFloatNear(output_data, kQuantizedTolerance)));
  }
  else
  {
    EXPECT_THAT(dequantize<T>(extractTensorData<T>(output_tensor), output_tensor.scale(),
                              output_tensor.zero_point()),
                ElementsAreArray(ArrayFloatNear(output_data, kQuantizedTolerance)));
  }
}

template <typename T> class Conv2DTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(Conv2DTest, DataTypes);

TYPED_TEST(Conv2DTest, TotalTest)
{
  Check<TypeParam>(/*input_shape=*/{1, 4, 3, 2}, /*filter_shape=*/{2, 2, 2, 2}, /*bias_shape=*/{2},
                   /*output_shape=*/{2, 1, 2, 2},
                   /*input_data=*/
                   {
                       1,  2,  3,  4,  5,  6,  // row = 0
                       7,  8,  9,  10, 11, 12, // row = 1
                       13, 14, 15, 16, 17, 18, // row = 2
                       19, 20, 21, 22, 23, 24, // row = 3
                   },
                   /*filter_shape=*/
                   {
                       1, 2, -3, -4, // out = 0, row = 0
                       -5, 6, -7, 8, // out = 1, row = 0
                       4, -2, 3, -1, // out = 0, row = 1
                       -8, -6, 7, 5, // out = 1, row = 1
                   },
                   /*bias_shape=*/{1, 2},
                   /*output_shape=*/
                   {
                       11, 16, 7, 20, // row = 0
                       0, 40, 0, 44,  // row = 1
                   });
}
} // namespace
} // namespace kernels
} // namespace luci_interpreter
