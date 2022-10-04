/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Dequantize.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class DequantizeTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(DequantizeTest, Uint8)
{
  std::vector<uint8_t> input_data{0, 1, 2, 3, 4, 251, 252, 253, 254, 255};

  std::vector<float> ref_output_data{-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64};

  Tensor input_tensor(loco::DataType::U8, {2, 5}, {{0.5}, {127}}, "");

  _memory_manager->allocate_memory(input_tensor);
  input_tensor.writeData(input_data.data(), input_data.size() * sizeof(uint8_t));

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Dequantize kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 5}));
}

TEST_F(DequantizeTest, Sint8)
{
  std::vector<int8_t> input_data{-128, -127, -126, -125, -124, 123, 124, 125, 126, 127};

  std::vector<float> ref_output_data{-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64};

  Tensor input_tensor(loco::DataType::S8, {2, 5}, {{0.5}, {-1}}, "");

  _memory_manager->allocate_memory(input_tensor);
  input_tensor.writeData(input_data.data(), input_data.size() * sizeof(int8_t));

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Dequantize kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 5}));
}

TEST_F(DequantizeTest, Sint16)
{
  std::vector<int16_t> input_data{-129, -126, -125, -124, -123, 124, 125, 126, 127, 131};

  std::vector<float> ref_output_data{-64.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 65.5};

  Tensor input_tensor(loco::DataType::S16, {2, 5}, {{0.5}, {0}}, "");

  _memory_manager->allocate_memory(input_tensor);
  input_tensor.writeData(input_data.data(), input_data.size() * sizeof(int16_t));

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Dequantize kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 5}));
}

TEST_F(DequantizeTest, InvalidInputType_NEG)
{
  std::vector<float> input_data{-129, -126, -125, -124, -123, 124, 125, 126, 127, 131};

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 5}, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Dequantize kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(DequantizeTest, InvalidOutputType_NEG)
{
  std::vector<int16_t> input_data{-129, -126, -125, -124, -123, 124, 125, 126, 127, 131};

  Tensor input_tensor(loco::DataType::S16, {2, 5}, {{0.5}, {0}}, "");

  _memory_manager->allocate_memory(input_tensor);
  input_tensor.writeData(input_data.data(), input_data.size() * sizeof(int16_t));

  Tensor output_tensor = makeOutputTensor(DataType::S8, /*scale*/ 0.5, /*zero_point*/ -1);

  Dequantize kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(DequantizeTest, InvalidInputZeroPoint_NEG)
{
  std::vector<float> input_data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  Tensor input_tensor =
    makeInputTensor<DataType::S16>({2, 5}, 0.5, -1, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Dequantize kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
