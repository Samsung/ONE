/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

// TODO enable it
#if 0
#include "kernels/ExpandDims.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class ExpandDimsTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(ExpandDimsTest, PositiveAxis)
{
  std::vector<int32_t> input_data{-1, 1, -2, 2};
  std::initializer_list<int32_t> input_shape = {2, 2};

  std::initializer_list<int32_t> axis_value = {0};

  Tensor input_tensor =
    makeInputTensor<DataType::S32>(input_shape, input_data, _memory_manager.get());
  Tensor axis_tensor = makeInputTensor<DataType::S32>({1}, axis_value, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S32);

  ExpandDims kernel(&input_tensor, &axis_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<int32_t>(output_tensor), ::testing::ElementsAreArray(input_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 2}));
}

TEST_F(ExpandDimsTest, NegAxis)
{
  std::vector<int32_t> input_data{-1, 1, -2, 2};
  std::initializer_list<int32_t> input_shape = {2, 2};

  std::initializer_list<int32_t> axis_value = {-1};

  Tensor input_tensor =
    makeInputTensor<DataType::S32>(input_shape, input_data, _memory_manager.get());
  Tensor axis_tensor = makeInputTensor<DataType::S32>({1}, axis_value, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S32);

  ExpandDims kernel(&input_tensor, &axis_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<int32_t>(output_tensor), ::testing::ElementsAreArray(input_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 2, 1}));
}

TEST_F(ExpandDimsTest, InvalidAxisType_NEG)
{
  std::vector<int32_t> input_data{-1, 1, -2, 2};
  std::initializer_list<int32_t> input_shape = {2, 2};

  std::initializer_list<float> axis_value = {1.0};

  Tensor input_tensor =
    makeInputTensor<DataType::S32>(input_shape, input_data, _memory_manager.get());
  Tensor axis_tensor = makeInputTensor<DataType::FLOAT32>({1}, axis_value, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S32);

  ExpandDims kernel(&input_tensor, &axis_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(ExpandDimsTest, InvalidAxisValue_NEG)
{
  std::vector<int32_t> input_data{-1, 1, -2, 2};
  std::initializer_list<int32_t> input_shape = {2, 2};

  std::initializer_list<int32_t> axis_value = {3};

  Tensor input_tensor =
    makeInputTensor<DataType::S32>(input_shape, input_data, _memory_manager.get());
  Tensor axis_tensor = makeInputTensor<DataType::S32>({1}, axis_value, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S32);

  ExpandDims kernel(&input_tensor, &axis_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
#endif
