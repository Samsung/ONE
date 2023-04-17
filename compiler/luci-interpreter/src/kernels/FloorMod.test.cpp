/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/FloorMod.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class FloorModTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(FloorModTest, FloatSimple)
{
  Shape input1_shape{1, 2, 2, 1};
  std::vector<float> input1_data{10.0, 9.0, 11.0, 3.0};

  Shape input2_shape = input1_shape;
  std::vector<float> input2_data{2.0, 2.0, 3.0, 4.0};

  std::vector<int32_t> ref_output_shape{1, 2, 2, 1};
  std::vector<float> ref_output_data{0.0, 1.0, 2.0, 3.0};

  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>(input1_shape, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::FLOAT32>(input2_shape, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FloorMod kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(FloorModTest, FloatNegativeValue)
{
  Shape input1_shape{1, 2, 2, 1};
  std::vector<float> input1_data{10.0, -9.0, -11.0, 7.0};

  Shape input2_shape = input1_shape;
  std::vector<float> input2_data{2.0, 2.0, -3.0, -4.0};

  std::vector<int32_t> ref_output_shape{1, 2, 2, 1};
  std::vector<float> ref_output_data{0.0, 1.0, -2.0, -1.0};

  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>(input1_shape, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::FLOAT32>(input2_shape, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FloorMod kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(FloorModTest, FloatBroadcast)
{
  Shape input1_shape{1, 2, 2, 1};
  std::vector<float> input1_data{
    10.0,
    -9.0,
    -11.0,
    7.0,
  };

  Shape input2_shape{1};
  std::vector<float> input2_data{-3.0};

  std::vector<int32_t> ref_output_shape{1, 2, 2, 1};
  std::vector<float> ref_output_data{-2.0, 0.0, -2.0, -2.0};

  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>(input1_shape, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::FLOAT32>(input2_shape, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FloorMod kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(FloorModTest, DivByZero_NEG)
{
  Shape shape{3};
  std::vector<float> input1_data{1, 0, -1};
  std::vector<float> input2_data{0, 0, 0};

  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>(shape, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::FLOAT32>(shape, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FloorMod kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);

  EXPECT_ANY_THROW(kernel.execute());
}

TEST_F(FloorModTest, Input_Output_Type_Mismatch_NEG)
{
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f}, _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  FloorMod kernel(&input1_tensor, &input2_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(FloorModTest, Input_Type_Mismatch_NEG)
{
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1}, _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::U8>({1}, {1}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FloorMod kernel(&input1_tensor, &input2_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
