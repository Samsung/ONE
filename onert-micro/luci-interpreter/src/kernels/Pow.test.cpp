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

#include "kernels/Pow.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class PowTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(PowTest, SimplePow)
{
  std::initializer_list<int32_t> base_shape = {1, 1, 3, 2};

  std::vector<float> input1_data{0.3f, 2.3f, 0.9f, 0.5f, 0.8f, 1.1f};
  std::vector<float> input2_data{0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f};
  std::vector<float> test_outputs{0.786f, 1.2838f, 1.043f, 0.7071f, 0.8f, 1.08956f};

  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>(base_shape, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::FLOAT32>(base_shape, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pow kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(test_outputs, 0.0001f));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(base_shape));
}

TEST_F(PowTest, FloatBroadcastPow)
{
  std::initializer_list<int32_t> input1_shape = {1, 3};
  std::initializer_list<int32_t> input2_shape = {3, 1};

  std::vector<float> input1_data{0.3f, 2.3f, 0.9f};
  std::vector<float> input2_data{0.2f, 0.3f, 0.4f};
  std::vector<float> test_outputs{0.786f,   1.18126f, 0.9791f, 0.6968f, 1.28386f,
                                  0.96888f, 0.6178f,  1.3953f, 0.9587f};

  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>(input1_shape, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::FLOAT32>(input2_shape, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pow kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(test_outputs, 0.0001f));
}

TEST_F(PowTest, IntPow)
{
  std::initializer_list<int32_t> base_shape = {1, 3};

  std::vector<int32_t> input_data{2, 3, 4};
  std::vector<int32_t> test_outputs{4, 27, 256};

  Tensor input1_tensor =
    makeInputTensor<DataType::S32>(base_shape, input_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::S32>(base_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S32);

  Pow kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<int32_t>(output_tensor), ::testing::ElementsAreArray(test_outputs));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(base_shape));
}

TEST_F(PowTest, Input_Output_Type_NEG)
{
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.0f}, _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.0f}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  Pow kernel(&input1_tensor, &input2_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(PowTest, Input_Type_Mismatch_NEG)
{
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.0f}, _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::S32>({1}, {4}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pow kernel(&input1_tensor, &input2_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(PowTest, Invalid_Input_Type_NEG)
{
  Tensor input1_tensor = makeInputTensor<DataType::S64>({1}, {1}, _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::S64>({1}, {1}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Pow kernel(&input1_tensor, &input2_tensor, &output_tensor);
  kernel.configure();
  EXPECT_ANY_THROW(kernel.execute());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
