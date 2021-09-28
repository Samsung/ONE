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

#include "kernels/Relu.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class ReluTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(ReluTest, FloatSimple)
{
  std::vector<float> input_data{
    0.0f, 1.0f,  3.0f,  // Row 1
    1.0f, -1.0f, -2.0f, // Row 2
  };

  std::vector<float> ref_output_data{
    0.0f, 1.0f, 3.0f, // Row 1
    1.0f, 0.0f, 0.0f, // Row 2
  };

  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3}, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Relu kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 3}));
}

TEST_F(ReluTest, Uint8Quantized)
{
  std::vector<float> input_data{
    0, -6, 2, 4, //
    3, -2, 7, 1, //
  };
  // Choose min / max in such a way that there are exactly 256 units to avoid rounding errors.
  const float f_min = (-128.0 / 128.0) * 8;
  const float f_max = (127.0 / 128.0) * 8;

  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(f_min, f_max);
  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 4, 1}, quant_param.first, quant_param.second, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  Relu kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray({128, 128, 160, 192, 176, 128, 240, 144}));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear({0, 0, 2, 4, 3, 0, 7, 1}));
}

TEST_F(ReluTest, Uint8Requantized)
{
  std::vector<float> input_data{
    0, -6, 2, 4, //
    3, -2, 7, 1, //
  };

  // Choose min / max in such a way that there are exactly 256 units to avoid rounding errors.
  const float in_min = (-128.0 / 128.0) * 8;
  const float in_max = (127.0 / 128.0) * 8;
  const float out_min = (0.0 / 256.0) * 8;
  const float out_max = (255.0 / 256.0) * 8;

  std::pair<float, int32_t> quant_input = quantizationParams<uint8_t>(in_min, in_max);
  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 4, 1}, quant_input.first, quant_input.second, input_data, _memory_manager.get());

  std::pair<float, int32_t> quant_output = quantizationParams<uint8_t>(out_min, out_max);
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_output.first, quant_output.second);

  Relu kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray({0, 0, 64, 128, 96, 0, 224, 32}));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear({0, 0, 2, 4, 3, 0, 7, 1}));
}

TEST_F(ReluTest, SInt16)
{
  std::vector<float> input_data{
    0, -6, 2, 4, //
    3, -2, 7, 1, //
  };
  std::vector<float> ref_output_data{
    0, 0, 2, 4, //
    3, 0, 7, 1, //
  };

  Tensor input_tensor =
    makeInputTensor<DataType::S16>({1, 2, 4, 1}, 0.5, 0, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.25, 0);

  Relu kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST_F(ReluTest, Input_Output_Type_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  Relu kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(ReluTest, Invalid_Input_Type_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::S64>({1}, {1}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Relu kernel(&input_tensor, &output_tensor);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  EXPECT_ANY_THROW(kernel.execute());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
