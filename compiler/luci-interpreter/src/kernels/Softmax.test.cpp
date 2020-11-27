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

#include "kernels/Softmax.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<float> input_data, std::initializer_list<float> output_data)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  SoftmaxParams params{};
  params.beta = 0.1;

  Softmax kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), FloatArrayNear(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), output_shape);
}

template <>
void Check<uint8_t>(std::initializer_list<int32_t> input_shape,
                    std::initializer_list<int32_t> output_shape,
                    std::initializer_list<float> input_data,
                    std::initializer_list<float> output_data)
{
  std::pair<float, int32_t> input_quant_param =
    quantizationParams<uint8_t>(std::min<float>(std::min<float>(input_data), 0.f),
                                std::max<float>(std::max<float>(input_data), 0.f));
  std::pair<float, int32_t> output_quant_param =
    quantizationParams<uint8_t>(std::min<float>(std::min<float>(output_data), 0.f),
                                std::max<float>(std::max<float>(output_data), 0.f));
  Tensor input_tensor = makeInputTensor<DataType::U8>(input_shape, input_quant_param.first,
                                                      input_quant_param.second, input_data);
  Tensor output_tensor =
    makeOutputTensor(DataType::U8, output_quant_param.first, output_quant_param.second);

  SoftmaxParams params{};
  params.beta = 0.1;

  Softmax kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(output_data, output_tensor.scale()));
}

template <typename T> class SoftmaxTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(SoftmaxTest, DataTypes);

TYPED_TEST(SoftmaxTest, Simple)
{
  Check<TypeParam>({2, 1, 2, 3}, {2, 1, 2, 3},
                   {
                     5, -9, 8,  //
                     -7, 2, -4, //
                     1, -2, 9,  //
                     3, -6, -1, //
                   },
                   {
                     0.38514, 0.09497, 0.51989, //
                     0.20792, 0.51141, 0.28067, //
                     0.25212, 0.18678, 0.56110, //
                     0.48149, 0.19576, 0.32275, //
                   });
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
