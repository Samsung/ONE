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
#include "kernels/L2Normalize.h"
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

  L2NormParams params{};
  params.activation = Activation::NONE;

  L2Normalize kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

template <>
void Check<uint8_t>(std::initializer_list<int32_t> input_shape,
                    std::initializer_list<int32_t> output_shape,
                    std::initializer_list<float> input_data,
                    std::initializer_list<float> output_data)
{
  std::pair<float, int32_t> quant_param =
      quantizationParams<uint8_t>(std::min(input_data) < 0 ? std::min(input_data) : 0.f,
                                  std::max(input_data) > 0 ? std::max(input_data) : 0.f);

  Tensor input_tensor =
      makeInputTensor<DataType::U8>(input_shape, quant_param.first, quant_param.second, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::U8, 1. / 128., 128);

  L2NormParams params{};
  params.activation = Activation::NONE;

  L2Normalize kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(output_data, output_tensor.scale()));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

template <typename T> class L2NormalizeTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(L2NormalizeTest, DataTypes);

TYPED_TEST(L2NormalizeTest, Simple)
{
  Check<TypeParam>({1, 1, 1, 6}, {1, 1, 1, 6}, {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1},
                   {-0.55, 0.3, 0.35, 0.6, -0.35, 0.05});
}

TEST(L2NormalizeTest, ActivationType_NEG)
{
  std::vector<float> input_data = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1};

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 1, 1, 6}, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  L2NormParams params{};
  params.activation = Activation::RELU6;

  L2Normalize kernel(&input_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(L2NormalizeTest, InvalidOutputQuantParam_NEG)
{
  std::vector<float> input_data = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1};

  Tensor input_tensor = makeInputTensor<DataType::U8>({1, 1, 1, 6}, 1. / 64., 127, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::U8, 1. / 64., 127);

  L2NormParams params{};
  params.activation = Activation::NONE;

  L2Normalize kernel(&input_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
