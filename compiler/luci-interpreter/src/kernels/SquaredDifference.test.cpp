/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/SquaredDifference.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/SimpleMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(SquaredDifferenceTest, Float)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();

  Shape input_shape{3, 1, 2};
  std::vector<float> input_data1{1.0, 0.0, -1.0, 11.0, -2.0, -1.44};
  std::vector<float> input_data2{-1.0, 0.0, 1.0, 12.0, -3.0, -1.43};
  Tensor input_tensor1 =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data1, memory_manager.get());
  Tensor input_tensor2 =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data2, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  SquaredDifference kernel(&input_tensor1, &input_tensor2, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{4.0, 0.0, 4.0, 1.0, 1.0, 0.0001};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
}

TEST(SquaredDifferenceTest, FloatBroadcast)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<SimpleMemoryManager>();

  Shape input_shape1{3, 1, 2};
  Shape input_shape2{1};
  std::vector<float> input_data1{1.0, 0.0, -1.0, 11.0, -2.0, -1.44};
  std::vector<float> input_data2{1.0};
  Tensor input_tensor1 =
    makeInputTensor<DataType::FLOAT32>(input_shape1, input_data1, memory_manager.get());
  Tensor input_tensor2 =
    makeInputTensor<DataType::FLOAT32>(input_shape2, input_data2, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  SquaredDifference kernel(&input_tensor1, &input_tensor2, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{0.0, 1.0, 4.0, 100.0, 9.0, 5.9536};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
