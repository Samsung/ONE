/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved
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

#include "kernels/Tile.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(TileTest, FloatMul12)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Shape input_shape{1, 3};
  std::vector<float> input_data{1.0f, 2.0f, 3.0f};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Shape mul_shape{2};
  std::vector<int32_t> mul_data{1, 2};
  Tensor mul_tensor = makeInputTensor<DataType::S32>(mul_shape, mul_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tile kernel(&input_tensor, &mul_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<int32_t> ref_output_shape{1, 6};
  std::vector<float> ref_output_data{1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(TileTest, FloatMul21)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Shape input_shape{1, 3};
  std::vector<float> input_data{1.0f, 2.0f, 3.0f};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Shape mul_shape{2};
  std::vector<int32_t> mul_data{2, 1};
  Tensor mul_tensor = makeInputTensor<DataType::S32>(mul_shape, mul_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tile kernel(&input_tensor, &mul_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<int32_t> ref_output_shape{2, 3};
  std::vector<float> ref_output_data{1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(TileTest, MultiplesShapeInvalid_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Shape input_shape{1, 3};
  std::vector<float> input_data{1.0f, 2.0f, 3.0f};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Shape mul_shape{3};
  std::vector<int32_t> mul_data{1, 2, 3};
  Tensor mul_tensor = makeInputTensor<DataType::S32>(mul_shape, mul_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tile kernel(&input_tensor, &mul_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(TileTest, MultiplesDTypeInvalid_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Shape input_shape{1, 3};
  std::vector<float> input_data{1.0f, 2.0f, 3.0f};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Shape mul_shape{2};
  std::vector<float> mul_data{1.0f, 2.0f};
  Tensor mul_tensor = makeInputTensor<DataType::FLOAT32>(mul_shape, mul_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Tile kernel(&input_tensor, &mul_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(TileTest, MultiplesDimInvalid_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Shape input_shape{1, 3};
  std::vector<float> input_data{1.0f, 2.0f, 3.0f};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, memory_manager.get());
  Shape mul_shape{3};
  std::vector<int32_t> mul_data{1, 2, 3};
  Tensor mul_tensor = makeInputTensor<DataType::S32>(mul_shape, mul_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S32);

  Tile kernel(&input_tensor, &mul_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
