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

#include "kernels/MaxPool2D.h"
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

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<float> input_data, std::initializer_list<float> output_data)
{
  float kQuantizedTolerance =
      GetTolerance(std::min<float>(input_data), std::max<float>(input_data));
  std::pair<float, int32_t> input_quant_param =
      quantizationParams<T>(std::min<float>(input_data), std::max<float>(input_data));
  Tensor input_tensor{getElementType<T>(),
                      input_shape,
                      {{input_quant_param.first}, {input_quant_param.second}},
                      ""};
  if (std::is_floating_point<T>::value)
  {
    input_tensor.writeData(input_data.begin(), input_data.size() * sizeof(T));
  }
  else
  {
    std::vector<T> quantized_input_value =
        quantize<T>(input_data, input_quant_param.first, input_quant_param.second);
    input_tensor.writeData(quantized_input_value.data(), quantized_input_value.size() * sizeof(T));
  }
  Tensor output_tensor =
      makeOutputTensor(getElementType<T>(), input_quant_param.first, input_quant_param.second);

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 3;
  params.stride_height = 1;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  MaxPool2D kernel(&input_tensor, &output_tensor, params);
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

template <typename T> class MaxPool2DTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(MaxPool2DTest, DataTypes);

TYPED_TEST(MaxPool2DTest, TotalTest)
{
  Check<TypeParam>(/*input_shape=*/{1, 3, 5, 1}, /*output_shape=*/{1, 2, 2, 1},
                   /*input_data=*/
                   {
                       1, -1, 0, -2, 2,    //
                       -7, -6, -5, -4, -3, //
                       5, 4, 3, 6, 7,      //
                   },
                   /*output_data=*/{
                       1, 2, //
                       5, 6, //
                   });
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
