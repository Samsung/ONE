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

#include "kernels/LeakyRelu.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<float> input_data, std::initializer_list<float> output_data,
           float alpha)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType element_type = getElementType<T>();
  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(element_type);

  LeakyReluParams params{};
  params.alpha = alpha;

  LeakyRelu kernel(&input_tensor, &output_tensor, params);

  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
}

template <>
void Check<uint8_t>(std::initializer_list<int32_t> input_shape,
                    std::initializer_list<int32_t> output_shape,
                    std::initializer_list<float> input_data,
                    std::initializer_list<float> output_data, float alpha)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  const float quantized_tolerance = getTolerance(-8, 127.f / 16.f, 255);
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-8, 127.f / 16.f);
  Tensor input_tensor = makeInputTensor<DataType::U8>(
    input_shape, quant_param.first, quant_param.second, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  LeakyReluParams params{};
  params.alpha = alpha;

  LeakyRelu kernel(&input_tensor, &output_tensor, params);

  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(output_data, quantized_tolerance));
}

template <typename T> class LeakReluTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_SUITE(LeakReluTest, DataTypes);

TYPED_TEST(LeakReluTest, Simple)
{
  Check<TypeParam>(/*input_shape=*/{2, 3}, /*output_shape=*/{2, 3},
                   /*input_data=*/
                   {
                     0.0f, 1.0f, 3.0f,   // Row 1
                     1.0f, -1.0f, -2.0f, // Row 2
                   },
                   /*output_data=*/
                   {
                     0.0f, 1.0f, 3.0f,   // Row 1
                     1.0f, -0.5f, -1.0f, // Row 2
                   },
                   /*alpha=*/0.5f);

  SUCCEED();
}

TEST(LeakReluTest, IvalidInputOutputType_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({2, 3},
                                                           {
                                                             0.0f, 1.0f, 3.0f,   // Row 1
                                                             1.0f, -1.0f, -2.0f, // Row 2
                                                           },
                                                           memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  LeakyReluParams params{};
  params.alpha = 0.5f;

  LeakyRelu kernel(&input_tensor, &output_tensor, params);

  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
