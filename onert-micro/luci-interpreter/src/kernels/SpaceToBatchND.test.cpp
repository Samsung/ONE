/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/SpaceToBatchND.h"
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
void Check(std::initializer_list<int32_t> input_shape,
           std::initializer_list<int32_t> block_shape_shape,
           std::initializer_list<int32_t> paddings_shape,
           std::initializer_list<int32_t> output_shape, std::initializer_list<float> input_data,
           std::initializer_list<int32_t> block_shape_data,
           std::initializer_list<int32_t> paddings_data, std::initializer_list<float> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType element_type = getElementType<T>();
  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());
  Tensor block_shape_tensor =
    makeInputTensor<DataType::S32>(block_shape_shape, block_shape_data, memory_manager.get());
  Tensor paddings_tensor =
    makeInputTensor<DataType::S32>(paddings_shape, paddings_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(element_type);

  SpaceToBatchND kernel(&input_tensor, &block_shape_tensor, &paddings_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), output_shape);
}

template <>
void Check<uint8_t>(
  std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> block_shape_shape,
  std::initializer_list<int32_t> paddings_shape, std::initializer_list<int32_t> output_shape,
  std::initializer_list<float> input_data, std::initializer_list<int32_t> block_shape_data,
  std::initializer_list<int32_t> paddings_data, std::initializer_list<float> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  std::pair<float, int32_t> input_quant_param =
    quantizationParams<uint8_t>(std::min(input_data), std::max(input_data));
  Tensor input_tensor =
    makeInputTensor<DataType::U8>(input_shape, input_quant_param.first, input_quant_param.second,
                                  input_data, memory_manager.get());
  Tensor block_shape_tensor =
    makeInputTensor<DataType::S32>(block_shape_shape, block_shape_data, memory_manager.get());
  Tensor paddings_tensor =
    makeInputTensor<DataType::S32>(paddings_shape, paddings_data, memory_manager.get());
  Tensor output_tensor =
    makeOutputTensor(DataType::U8, input_quant_param.first, input_quant_param.second);

  SpaceToBatchND kernel(&input_tensor, &block_shape_tensor, &paddings_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(output_data, output_tensor.scale()));
  EXPECT_THAT(extractTensorShape(output_tensor), output_shape);
}

template <typename T> class SpaceToBatchNDTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_SUITE(SpaceToBatchNDTest, DataTypes);

TYPED_TEST(SpaceToBatchNDTest, Simple)
{
  Check<TypeParam>(/*input_shape=*/{1, 5, 2, 1}, /*block_shape_shape=*/{2},
                   /*paddings_shape=*/{2, 2},
                   /*output_shape=*/{6, 2, 2, 1},
                   /*input_data=*/{-1.0, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8, -0.9, 1.0},
                   /*block_shape_data=*/{3, 2}, /*paddings_data=*/{1, 0, 2, 0},
                   /*output_data=*/{0, 0,   0, -0.5, 0, 0,    0, 0.6,  0, -1.0, 0, -0.7,
                                    0, 0.2, 0, 0.8,  0, -0.3, 0, -0.9, 0, 0.4,  0, 1.0});
}

TEST(SpaceToBatchNDTest, Invalid_Shape_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(
    {1, 3, 3, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, memory_manager.get());
  Tensor block_shape_tensor = makeInputTensor<DataType::S32>({2}, {2, 2}, memory_manager.get());
  Tensor paddings_tensor =
    makeInputTensor<DataType::S32>({2, 2}, {0, 0, 0, 0}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  SpaceToBatchND kernel(&input_tensor, &block_shape_tensor, &paddings_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
