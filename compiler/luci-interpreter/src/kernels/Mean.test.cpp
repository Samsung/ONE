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

#include "kernels/Mean.h"
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
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> axis_shape,
           std::initializer_list<int32_t> output_shape, std::initializer_list<float> input_data,
           std::initializer_list<int32_t> axis_data, std::initializer_list<float> output_data,
           bool keep_dims)
{
  float kQuantizedTolerance = GetTolerance(0.0f, std::max<float>(input_data));
  std::pair<float, int32_t> input_quant_param =
      quantizationParams<T>(0.0f, std::max<float>(input_data));

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
  Tensor axis_tensor{DataType::S32, axis_shape, {}, ""};
  axis_tensor.writeData(axis_data.begin(), axis_data.size() * sizeof(int32_t));
  Tensor output_tensor =
      makeOutputTensor(getElementType<T>(), input_quant_param.first, input_quant_param.second);

  ReducerParams params{};
  params.keep_dims = keep_dims;

  Mean kernel(&input_tensor, &axis_tensor, &output_tensor, params);
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

template <typename T> class MeanTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(MeanTest, DataTypes);

TYPED_TEST(MeanTest, TotalTest)
{
  Check<TypeParam>(/*input_shape=*/{4, 3, 2}, /*axis_shape=*/{2}, /*output_shape=*/{1, 3, 1},
                   /*input_data=*/{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,   //
                                   7.0,  8.0,  9.0,  10.0, 11.0, 12.0,  //
                                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0,  //
                                   19.0, 20.0, 21.0, 22.0, 23.0, 24.0}, //
                   /*axis_data=*/{0, 2},
                   /*output_data=*/{10.5, 12.5, 14.5}, /*keep_dims=*/true);
  Check<TypeParam>(/*input_shape=*/{2, 2, 3, 2}, /*axis_shape=*/{2}, /*output_shape=*/{2, 1, 1, 2},
                   /*input_data=*/{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,   //
                                   7.0,  8.0,  9.0,  10.0, 11.0, 12.0,  //
                                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0,  //
                                   19.0, 20.0, 21.0, 22.0, 23.0, 24.0}, //
                   /*axis_data=*/{1, 2},
                   /*output_data=*/{6, 7, 18, 19}, /*keep_dims=*/true);
  Check<TypeParam>(/*input_shape=*/{4, 3, 2}, /*axis_shape=*/{4}, /*output_shape=*/{2},
                   /*input_data=*/{1.0,  2.0,  3.0,  4.0,  5.0,  6.0,   //
                                   7.0,  8.0,  9.0,  10.0, 11.0, 12.0,  //
                                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0,  //
                                   19.0, 20.0, 21.0, 22.0, 23.0, 24.0}, //
                   /*axis_data=*/{1, 0, -3, -3},
                   /*output_data=*/{12, 13}, /*keep_dims=*/false);
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
