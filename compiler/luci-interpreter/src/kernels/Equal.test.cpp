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

#include "kernels/Equal.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/SimpleMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class EqualTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<SimpleMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(EqualTest, FloatSimple)
{
  std::vector<float> x_data{
    0.5, 0.7, 0.9, // Row 1
    1,   0,   -1,  // Row 2
  };

  std::vector<float> y_data{
    0.9, 0.7, 0.5, // Row 1
    -1,  0,   1,   // Row 2
  };

  std::vector<bool> ref_output_data{
    false, true, false, // Row 1
    false, true, false, // Row 2
  };

  Tensor x_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, x_data, _memory_manager.get());
  Tensor y_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, y_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  Equal kernel(&x_tensor, &y_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<bool>(output_tensor), ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 3}));
}

TEST_F(EqualTest, FloatBroardcast)
{
  std::vector<float> x_data{
    0.5, 0.7, 0.9, // Row 1
    1,   0,   -1,  // Row 2
    -1,  0,   1,   // Row 3
    0.9, 0.7, 0.5, // Row 4
  };

  std::vector<float> y_data{
    0.9, 0.7, 0.5, // Row 1
  };

  std::vector<bool> ref_output_data{
    false, true,  false, // Row 1
    false, false, false, // Row 2
    false, false, false, // Row 3
    true,  true,  true,  // Row 4
  };

  Tensor x_tensor = makeInputTensor<DataType::FLOAT32>({4, 3}, x_data, _memory_manager.get());
  Tensor y_tensor = makeInputTensor<DataType::FLOAT32>({1, 3}, y_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  Equal kernel(&x_tensor, &y_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<bool>(output_tensor), ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({4, 3}));
}

// Choose min / max in such a way that there are exactly 256 units to avoid rounding errors.
const float F_MIN = -128.0 / 128.0;
const float F_MAX = 127.0 / 128.0;

TEST_F(EqualTest, Uint8Quantized)
{
  std::vector<float> x_data{
    0.5, 0.5, 0.7,  0.9, // Row 1
    1,   0,   0.05, -1,  // Row 2
  };

  std::vector<float> y_data{
    0.9, 0.5, 0.55, 0.5, // Row 1
    -1,  0,   0.05, 1,   // Row 2
  };

  std::vector<bool> ref_output_data{
    false, true, false, false, // Row 1
    false, true, true,  false, // Row 2
  };

  std::pair<float, int32_t> x_quant_param = quantizationParams<uint8_t>(F_MIN, F_MAX);
  Tensor x_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 4, 1}, x_quant_param.first, x_quant_param.second, x_data, _memory_manager.get());

  std::pair<float, int32_t> y_quant_param = quantizationParams<uint8_t>(F_MIN * 2, F_MAX * 2);
  Tensor y_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 4, 1}, y_quant_param.first, y_quant_param.second, y_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  Equal kernel(&x_tensor, &y_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(extractTensorData<bool>(output_tensor), ::testing::ElementsAreArray(ref_output_data));
}

TEST_F(EqualTest, Uint8QuantizedBroadcast)
{
  std::vector<float> x_data{
    0.4,  -0.8, 0.7,  0.3, // Row 1
    -0.5, 0.1,  0,    0.5, // Row 2
    1,    0,    0.05, -1,  // Row 3
    -1,   0.05, 0,    1,   // Row 4
  };

  std::vector<float> y_data{
    -1, 0.05, 0, 1, // Row 1
  };

  std::vector<bool> ref_output_data{
    false, false, false, false, // Row 1
    false, false, true,  false, // Row 2
    false, false, false, false, // Row 3
    true,  true,  true,  true,  // Row 4
  };

  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(F_MIN, F_MAX);
  Tensor x_tensor = makeInputTensor<DataType::U8>(
    {1, 4, 4, 1}, quant_param.first, quant_param.second, x_data, _memory_manager.get());
  Tensor y_tensor = makeInputTensor<DataType::U8>(
    {1, 1, 4, 1}, quant_param.first, quant_param.second, y_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  Equal kernel(&x_tensor, &y_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 4, 4, 1}));
  EXPECT_THAT(extractTensorData<bool>(output_tensor), ::testing::ElementsAreArray(ref_output_data));
}

TEST_F(EqualTest, Input_Type_Mismatch_NEG)
{
  Tensor x_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f}, _memory_manager.get());
  Tensor y_tensor = makeInputTensor<DataType::U8>({1}, {1}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::BOOL);

  Equal kernel(&x_tensor, &y_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(EqualTest, Input_Output_Type_NEG)
{
  Tensor x_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f}, _memory_manager.get());
  Tensor y_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Equal kernel(&x_tensor, &y_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
