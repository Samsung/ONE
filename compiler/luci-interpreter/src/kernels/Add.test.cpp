/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/Add.h"
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

float getMinValue(std::initializer_list<float> data)
{
  return 0 > std::min<float>(data) ? std::min<float>(data) : 0;
}

float getMaxValue(std::initializer_list<float> data)
{
  return 0 < std::max<float>(data) ? std::max<float>(data) : 0;
}

template <typename T>
void Check(std::initializer_list<int32_t> input1_shape, std::initializer_list<int32_t> input2_shape,
           std::initializer_list<int32_t> output_shape, std::initializer_list<float> input1_data,
           std::initializer_list<float> input2_data, std::initializer_list<float> output_data)
{
  float kQuantizedTolerance = GetTolerance(getMinValue(input1_data) + getMinValue(input2_data),
                                           getMaxValue(input1_data) + getMaxValue(input2_data));
  std::pair<float, int32_t> input1_quant_param =
      quantizationParams<T>(getMinValue(input1_data), getMaxValue(input1_data));
  std::pair<float, int32_t> input2_quant_param =
      quantizationParams<T>(getMinValue(input2_data), getMaxValue(input2_data));
  Tensor input1_tensor{getElementType<T>(),
                       input1_shape,
                       {{input1_quant_param.first}, {input1_quant_param.second}},
                       ""};
  Tensor input2_tensor{getElementType<T>(),
                       input2_shape,
                       {{input2_quant_param.first}, {input2_quant_param.second}},
                       ""};
  if (std::is_floating_point<T>::value)
  {
    input1_tensor.writeData(input1_data.begin(), input1_data.size() * sizeof(T));
    input2_tensor.writeData(input2_data.begin(), input2_data.size() * sizeof(T));
  }
  else
  {
    std::vector<T> quantized_input1_value =
        quantize<T>(input1_data, input1_quant_param.first, input1_quant_param.second);
    std::vector<T> quantized_input2_value =
        quantize<T>(input2_data, input2_quant_param.first, input2_quant_param.second);
    input1_tensor.writeData(quantized_input1_value.data(),
                            quantized_input1_value.size() * sizeof(T));
    input2_tensor.writeData(quantized_input2_value.data(),
                            quantized_input2_value.size() * sizeof(T));
  }
  std::pair<float, int32_t> output_quant_param =
      quantizationParams<T>(getMinValue(input1_data) + getMinValue(input2_data),
                            getMaxValue(input1_data) + getMaxValue(input2_data));
  Tensor output_tensor =
      makeOutputTensor(getElementType<T>(), output_quant_param.first, output_quant_param.second);

  AddParams params{};
  params.activation = Activation::RELU;

  Add kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  if (std::is_floating_point<T>::value)
  {
    EXPECT_THAT(extractTensorData<T>(output_tensor), ElementsAreArray(ArrayFloatNear(output_data)));
  }
  else
  {
    EXPECT_THAT(dequantize<T>(extractTensorData<T>(output_tensor), output_tensor.scale(),
                              output_tensor.zero_point()),
                ElementsAreArray(ArrayFloatNear(output_data, kQuantizedTolerance)));
  }
}

template <typename T> class AddTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(AddTest, DataTypes);

TYPED_TEST(AddTest, TotalTest)
{
  Check<TypeParam>(
      /*input1_shape=*/{2, 3, 1, 2}, /*input2_shape=*/{1, 1, 3, 2}, /*output_shape=*/{2, 3, 3, 2},
      /*input1_data=*/{-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f, 2.8f, -1.6f, 0.0f, 0.7f, -2.2f},
      /*input2_data=*/{0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f},
      /*output_data=*/{0.0f, 2.6f, 0.0f, 2.8f, 0.7f, 3.2f, 1.1f, 0.8f, 0.5f, 1.0f, 1.9f, 1.4f,
                       1.0f, 0.0f, 0.4f, 0.0f, 1.8f, 0.0f, 1.4f, 3.1f, 0.8f, 3.3f, 2.2f, 3.7f,
                       0.0f, 0.3f, 0.0f, 0.5f, 0.0f, 0.9f, 0.9f, 0.0f, 0.3f, 0.0f, 1.7f, 0.0f});
  Check<TypeParam>(
      /*input1_shape=*/{2, 3, 1, 2}, /*input2_shape=*/{1, 3, 1, 2}, /*output_shape=*/{2, 3, 1, 2},
      /*input1_data=*/{-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f, 2.8f, -1.6f, 0.0f, 0.7f, -2.2f},
      /*input2_data=*/{0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f},
      /*output_data=*/{0.0f, 2.6f, 0.5f, 1.0f, 1.8f, 0.0f, 1.4f, 3.1f, 0.0f, 0.5f, 1.7f, 0.0f});
  Check<TypeParam>(
      /*input1_shape=*/{2, 3, 1, 2}, /*input2_shape=*/{2, 1, 3, 1}, /*output_shape=*/{2, 3, 3, 2},
      /*input1_data=*/{-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f, 2.8f, -1.6f, 0.0f, 0.7f, -2.2f},
      /*input2_data=*/{0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f},
      /*output_data=*/{0.0f, 2.5f, 0.0f, 2.6f, 0.0f, 1.9f, 1.1f, 0.7f, 1.2f, 0.8f, 0.5f, 0.1f,
                       1.0f, 0.0f, 1.1f, 0.0f, 0.4f, 0.0f, 1.7f, 3.3f, 2.2f, 3.8f, 2.1f, 3.7f,
                       0.0f, 0.5f, 0.0f, 1.0f, 0.0f, 0.9f, 1.2f, 0.0f, 1.7f, 0.0f, 1.6f, 0.0f});
  Check<TypeParam>(
      /*input1_shape=*/{2, 3, 1, 2}, /*input2_shape=*/{2, 3, 1, 1}, /*output_shape=*/{2, 3, 1, 2},
      /*input1_data=*/{-0.3f, 2.3f, 0.9f, 0.5f, 0.8f, -1.1f, 1.2f, 2.8f, -1.6f, 0.0f, 0.7f, -2.2f},
      /*input2_data=*/{0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f},
      /*output_data=*/{0.0f, 2.5f, 1.2f, 0.8f, 0.4f, 0.0f, 1.7f, 3.3f, 0.0f, 1.0f, 1.6f, 0.0f});
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
