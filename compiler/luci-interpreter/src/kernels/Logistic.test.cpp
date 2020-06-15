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

#include "kernels/Logistic.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;
const float kQuantizedTolerance = 2 * (1. / 256);

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<float> input_data, std::initializer_list<float> output_data,
           DataType element_type)
{
  std::pair<float, int32_t> input_quant_params =
      quantizationParams<T>(std::min(input_data), std::max(input_data));
  Tensor input_tensor{
      element_type, input_shape, {{input_quant_params.first}, {input_quant_params.second}}, ""};
  if (element_type == DataType::FLOAT32)
  {
    input_tensor.writeData(input_data.begin(), input_data.size() * sizeof(T));
  }
  else
  {
    std::vector<T> quantized_input_value =
        quantize<T>(input_data, input_quant_params.first, input_quant_params.second);
    input_tensor.writeData(quantized_input_value.data(), quantized_input_value.size() * sizeof(T));
  }
  Tensor output_tensor = makeOutputTensor(element_type, 1. / 256., 0);

  Logistic kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  if (element_type == DataType::FLOAT32)
  {
    EXPECT_THAT(extractTensorData<T>(output_tensor),
                ElementsAreArray(ArrayFloatNear(output_data, kQuantizedTolerance)));
  }
  else
  {
    EXPECT_THAT(dequantize(extractTensorData<T>(output_tensor), output_tensor.scale(),
                           output_tensor.zero_point()),
                ElementsAreArray(ArrayFloatNear(output_data, kQuantizedTolerance)));
  }
}

template <typename T> class LogisticTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(LogisticTest, DataTypes);

TYPED_TEST(LogisticTest, TotalTest)
{
  Check<TypeParam>(/*input_shape=*/{1, 6, 4, 1}, /*output_shape=*/{1, 6, 4, 1},
                   /*input_data=*/
                   {
                       0, -6, 2,  4, //
                       3, -2, 10, 1, //
                       0, -6, 2,  4, //
                       3, -2, 10, 1, //
                       0, -6, 2,  4, //
                       3, -2, 10, 1, //
                   },
                   /*output_data=*/
                   {
                       0.5,      0.002473, 0.880797, 0.982014, //
                       0.952574, 0.119203, 0.999955, 0.731059, //
                       0.5,      0.002473, 0.880797, 0.982014, //
                       0.952574, 0.119203, 0.999955, 0.731059, //
                       0.5,      0.002473, 0.880797, 0.982014, //
                       0.952574, 0.119203, 0.999955, 0.731059, //
                   },
                   getElementType<TypeParam>());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
