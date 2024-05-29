/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/BroadcastTo.h"
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
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> shape_shape,
           std::initializer_list<int32_t> output_shape, std::initializer_list<float> input_data,
           std::initializer_list<T> shape_data, std::initializer_list<float> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType element_type = DataType::FLOAT32;
  constexpr DataType shape_type = getElementType<T>();

  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());
  Tensor shape_tensor = makeInputTensor<shape_type>(shape_shape, shape_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(element_type);

  BroadcastTo kernel(&input_tensor, &shape_tensor, &output_tensor);

  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(output_data));
}

template <typename T>
void Check_bool(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> shape_shape,
           std::initializer_list<int32_t> output_shape, std::initializer_list<uint8_t> input_data,
           std::initializer_list<T> shape_data, std::initializer_list<uint8_t> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType element_type = DataType::BOOL;
  constexpr DataType shape_type = getElementType<T>();

  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());
  Tensor shape_tensor = makeInputTensor<shape_type>(shape_shape, shape_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(element_type);

  BroadcastTo kernel(&input_tensor, &shape_tensor, &output_tensor);

  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), output_shape);
}

class BroadcastToTest : public ::testing::Test
{
};

TEST_F(BroadcastToTest, SimpleS32)
{
  Check<int32_t>(/*input_shape*/ {1, 3}, /*shape_shape*/ {2}, /*output_shape*/ {2, 3},
                 /*input_data*/
                 {1, 2, 3},
                 /*shape_data*/
                 {2, 3},
                 /*output_data*/
                 {
                   1, 2, 3, // Row 1
                   1, 2, 3, // Row 2
                 });
  SUCCEED();
}

TEST_F(BroadcastToTest, SimpleS64)
{
  Check<int64_t>(/*input_shape*/ {1, 3}, /*shape_shape*/ {2}, /*output_shape*/ {2, 3},
                 /*input_data*/
                 {1, 2, 3},
                 /*shape_data*/
                 {2, 3},
                 /*output_data*/
                 {
                   1, 2, 3, // Row 1
                   1, 2, 3, // Row 2
                 });
  SUCCEED();
}

TEST_F(BroadcastToTest, SimpleBool)
{
  Check_bool<int32_t>(/*input_shape*/ {1, 3}, /*shape_shape*/ {2}, /*output_shape*/ {2, 3},
                 /*input_data*/
                 {true, false, true},
                 /*shape_data*/
                 {2, 3},
                 /*output_data*/
                 {
                   true, false, true, // Row 1
                   true, false, true, // Row 2
                 });
  SUCCEED();
}

TEST_F(BroadcastToTest, DifferentInOutType_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 3}, {1, 2, 3}, memory_manager.get());
  Tensor shape_tensor = makeInputTensor<DataType::S32>({2}, {2, 3}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  BroadcastTo kernel(&input_tensor, &shape_tensor, &output_tensor);

  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(BroadcastToTest, BroadcastAble_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3}, {1, 2, 3, 1, 2, 3}, memory_manager.get());
  Tensor shape_tensor = makeInputTensor<DataType::S32>({2}, {2, 6}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  BroadcastTo kernel(&input_tensor, &shape_tensor, &output_tensor);

  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace

} // namespace kernels
} // namespace luci_interpreter
