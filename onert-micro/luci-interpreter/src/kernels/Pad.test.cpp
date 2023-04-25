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

#include "kernels/Pad.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

float GetTolerance(float min, float max) { return (max - min) / 255.0; }

TEST(Pad, Uint8)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-1.0f, 1.0f);
  std::vector<float> input_data{-0.8, 0.2, 0.9, 0.7, 0.1, -0.3};
  std::vector<int32_t> paddings_data{0, 0, 0, 2, 1, 3, 0, 0};
  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 3, 1}, quant_param.first, quant_param.second, input_data, memory_manager.get());
  Tensor paddings_tensor =
    makeInputTensor<DataType::S32>({4, 2}, paddings_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  Pad kernel(&input_tensor, &paddings_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{0, -0.8, 0.2, 0.9, 0, 0, 0, 0, 0.7, 0.1, -0.3, 0, 0, 0,
                                     0, 0,    0,   0,   0, 0, 0, 0, 0,   0,   0,    0, 0, 0};
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(ref_output_data, kQuantizedTolerance));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 4, 7, 1}));
}

TEST(Pad, Int8)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::pair<float, int32_t> quant_param = quantizationParams<int8_t>(-1.0f, 1.0f);
  std::vector<float> input_data{-0.2, 0.4, 0.5, -0.7, -0.1, -0.9, 0.7, 0.1, 0.2};
  std::vector<int32_t> paddings_data{0, 0, 1, 2, 2, 1, 0, 0};
  Tensor input_tensor = makeInputTensor<DataType::S8>(
    {1, 3, 3, 1}, quant_param.first, quant_param.second, input_data, memory_manager.get());
  Tensor paddings_tensor =
    makeInputTensor<DataType::S32>({4, 2}, paddings_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S8, quant_param.first, quant_param.second);

  Pad kernel(&input_tensor, &paddings_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{0, 0, 0,    0,    0,    0, 0, 0, -0.2, 0.4, 0.5, 0,
                                     0, 0, -0.7, -0.1, -0.9, 0, 0, 0, 0.7,  0.1, 0.2, 0,
                                     0, 0, 0,    0,    0,    0, 0, 0, 0,    0,   0,   0};
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear(ref_output_data, kQuantizedTolerance));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 6, 6, 1}));
}

TEST(Pad, Float)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  std::vector<float> input_data{1, 2, 3, 4, 5, 6};
  std::vector<int32_t> paddings_data{1, 0, 0, 2, 0, 3, 0, 0};
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 2, 3, 1}, input_data, memory_manager.get());
  Tensor paddings_tensor =
    makeInputTensor<DataType::S32>({4, 2}, paddings_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pad kernel(&input_tensor, &paddings_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 4, 5,
                                     6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::initializer_list<int32_t> ref_output_shape{2, 4, 6, 1};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
