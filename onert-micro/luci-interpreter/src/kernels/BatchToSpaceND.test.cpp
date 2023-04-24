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

#include "kernels/BatchToSpaceND.h"
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
           std::initializer_list<int32_t> crops_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<T> input_data, std::initializer_list<int32_t> block_shape_data,
           std::initializer_list<int32_t> crops_data, std::initializer_list<T> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType element_type = getElementType<T>();
  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());
  Tensor block_shape_tensor =
    makeInputTensor<DataType::S32>(block_shape_shape, block_shape_data, memory_manager.get());
  Tensor crops_tensor =
    makeInputTensor<DataType::S32>(crops_shape, crops_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(element_type);

  BatchToSpaceND kernel(&input_tensor, &block_shape_tensor, &crops_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), output_shape);
}

template <typename T> class BatchToSpaceNDTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_SUITE(BatchToSpaceNDTest, DataTypes);

TYPED_TEST(BatchToSpaceNDTest, Simple)
{
  Check<TypeParam>(/*input_shape=*/{4, 2, 2, 1}, /*block_shape_shape=*/{2}, /*crops_shape=*/{2, 2},
                   /*output_shape=*/{1, 4, 4, 1},
                   /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                   /*block_shape_data=*/{2, 2}, /*crops_data=*/{0, 0, 0, 0},
                   /*output_data=*/{1, 5, 2, 6, 9, 13, 10, 14, 3, 7, 4, 8, 11, 15, 12, 16});
}

TEST(BatchToSpaceNDTest, Invalid_Shape_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(
    {3, 2, 2, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, memory_manager.get());
  Tensor block_shape_tensor = makeInputTensor<DataType::S32>({2}, {2, 2}, memory_manager.get());
  Tensor crops_tensor = makeInputTensor<DataType::S32>({2, 2}, {0, 0, 0, 0}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  BatchToSpaceND kernel(&input_tensor, &block_shape_tensor, &crops_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(BatchToSpaceNDTest, Invalid_Crops_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(
    {4, 2, 2, 1}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, memory_manager.get());
  Tensor block_shape_tensor = makeInputTensor<DataType::S32>({2}, {2, 2}, memory_manager.get());
  Tensor crops_tensor = makeInputTensor<DataType::S32>({2, 2}, {0, 0, -1, 0}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  BatchToSpaceND kernel(&input_tensor, &block_shape_tensor, &crops_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
