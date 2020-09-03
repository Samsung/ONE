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

#include "kernels/FullyConnected.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

// Returns the corresponding Bias DataType given the type T(Input Tensor Type).
template <typename T> constexpr DataType getBiasType()
{
  if (std::is_same<T, float>::value)
    return DataType::FLOAT32;
  if (std::is_same<T, uint8_t>::value)
    return DataType::S32;
  return DataType::Unknown;
}

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> weights_shape,
           std::initializer_list<int32_t> bias_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<float> input_data, std::initializer_list<float> weights_data,
           std::initializer_list<float> bias_data, std::initializer_list<float> output_data)
{
  float kQuantizedTolerance = getTolerance(-127, 128, 255);
  std::pair<float, int32_t> input_quant_param = quantizationParams<T>(-63.5, 64);
  std::pair<float, int32_t> output_quant_param = quantizationParams<T>(-127, 128);
  Tensor input_tensor{getElementType<T>(),
                      input_shape,
                      {{input_quant_param.first}, {input_quant_param.second}},
                      ""};
  Tensor weights_tensor{getElementType<T>(),
                        weights_shape,
                        {{input_quant_param.first}, {input_quant_param.second}},
                        ""};
  Tensor bias_tensor{
      getBiasType<T>(), bias_shape, {{input_quant_param.first * input_quant_param.first}, {0}}, ""};
  if (std::is_floating_point<T>::value)
  {
    input_tensor.writeData(input_data.begin(), input_data.size() * sizeof(T));
    weights_tensor.writeData(weights_data.begin(), weights_data.size() * sizeof(T));
    bias_tensor.writeData(bias_data.begin(), bias_data.size() * sizeof(T));
  }
  else
  {
    std::vector<T> quantized_input_value =
        quantize<T>(input_data, input_quant_param.first, input_quant_param.second);
    std::vector<T> quantized_weights_value =
        quantize<T>(weights_data, input_quant_param.first, input_quant_param.second);
    std::vector<int32_t> quantized_bias_value =
        quantize<int32_t>(bias_data, bias_tensor.scale(), bias_tensor.zero_point());
    input_tensor.writeData(quantized_input_value.data(), quantized_input_value.size() * sizeof(T));
    weights_tensor.writeData(quantized_weights_value.data(),
                             quantized_weights_value.size() * sizeof(T));
    bias_tensor.writeData(quantized_bias_value.data(),
                          quantized_bias_value.size() * sizeof(int32_t));
  }

  Tensor output_tensor =
      makeOutputTensor(getElementType<T>(), output_quant_param.first, output_quant_param.second);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor, params);
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
  EXPECT_THAT(extractTensorShape(output_tensor), output_shape);
}

template <typename T> class FullyConnectedTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(FullyConnectedTest, DataTypes);

TYPED_TEST(FullyConnectedTest, Simple)
{
  Check<TypeParam>({3, 2, 2, 1}, {3, 6}, {3}, {2, 3},
                   {
                       -3, -5, 5, 4, 9, -2,  // batch = 0
                       -3, -2, -4, 9, -8, 1, // batch = 1
                   },
                   {
                       -3, -7, 4, -4, -6, 4, // unit = 0
                       3, 5, 2, 3, -3, -8,   // unit = 1
                       -3, 7, 4, 9, 0, -5,   // unit = 2
                   },
                   {-1, -5, -8}, {
                                     0, 0, 32,   // batch = 0
                                     22, 11, 47, // batch = 1
                                 });
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
