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

#include "kernels/Maximum.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/SimpleMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class MaximumTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<SimpleMManager>(); }

  std::unique_ptr<MManager> _memory_manager;
};

TEST_F(MaximumTest, Float)
{
  Shape input_shape{3, 1, 2};
  std::vector<float> input_data1{1.0, 0.0, -1.0, 11.0, -2.0, -1.44};
  std::vector<float> input_data2{-1.0, 0.0, 1.0, 12.0, -3.0, -1.43};
  Tensor input_tensor1 =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data1, _memory_manager.get());
  Tensor input_tensor2 =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data2, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Maximum kernel(&input_tensor1, &input_tensor2, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{1.0, 0.0, 1.0, 12.0, -2.0, -1.43};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
}

TEST_F(MaximumTest, Uint8)
{
  Shape input_shape{3, 1, 2};
  std::vector<uint8_t> input_data1{1, 0, 2, 11, 2, 23};
  std::vector<uint8_t> input_data2{0, 0, 1, 12, 255, 1};
  Tensor input_tensor1 =
    makeInputTensor<DataType::U8>(input_shape, input_data1, _memory_manager.get());
  Tensor input_tensor2 =
    makeInputTensor<DataType::U8>(input_shape, input_data2, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  Maximum kernel(&input_tensor1, &input_tensor2, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  std::vector<int32_t> ref_output_shape{2, 4};
  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray({1, 0, 2, 12, 255, 23}));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
