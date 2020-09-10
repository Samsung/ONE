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

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(ReluTest, FloatSimple)
{
  std::vector<float> input_data{
      0.0f, 1.0f,  3.0f,  // Row 1
      1.0f, -1.0f, -2.0f, // Row 2
  };

  std::vector<float> ref_output_data{
      0.0f, 1.0f, 3.0f, // Row 1
      1.0f, 0.0f, 0.0f, // Row 2
  };

  Tensor input_tensor{DataType::FLOAT32, {2, 3}, {}, ""};
  input_tensor.writeData(input_data.data(), input_data.size() * sizeof(float));

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Relu kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 3}));
}

float GetTolerance(float min, float max) { return (max - min) / 255.0; }

TEST(ReluTest, Uint8Dequantized)
{
  std::vector<float> input_data{-0.8f, 0.2f, 0.9f, 0.7f, 0.1f, -0.4f};
  std::vector<float> ref_output_data{0.0f, 0.2f, 0.9f, 0.7f, 0.1f, 0.0f};

  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-1.0f, 1.0f);

  Tensor input_tensor{DataType::U8, {1, 2, 3, 1}, {{quant_param.first}, {quant_param.second}}, ""};
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  std::vector<uint8_t> quantize_input =
      quantize<uint8_t>(input_data, quant_param.first, quant_param.second);
  input_tensor.writeData(quantize_input.data(), quantize_input.size() * sizeof(uint8_t));

  Relu kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(dequantize(extractTensorData<uint8_t>(output_tensor), output_tensor.scale(),
                         output_tensor.zero_point()),
              ElementsAreArray(ArrayFloatNear(ref_output_data, kQuantizedTolerance)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 3, 1}));
}

TEST(ReluTest, Uint8Quantized)
{
  const float kMin = -1.f;
  const float kMax = 1.f;

  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(kMin, kMax);
  Tensor input_tensor{DataType::U8, {1, 2, 4, 1}, {{quant_param.first}, {quant_param.second}}, ""};
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  std::vector<uint8_t> quantize_input{
      127, 67, 160, 192, //
      176, 85, 240, 144, //
  };
  input_tensor.writeData(quantize_input.data(), quantize_input.size() * sizeof(uint8_t));

  Relu kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray({127, 127, 160, 192, 176, 127, 240, 144}));
}

TEST(ReluTest, Uint8Requantized)
{
  const float kMin = -1.f;
  const float kMinOut = 0.f;
  const float kMax = 1.f;

  std::pair<float, int32_t> quant_input = quantizationParams<uint8_t>(kMin, kMax);
  Tensor input_tensor{DataType::U8, {1, 2, 4, 1}, {{quant_input.first}, {quant_input.second}}, ""};

  std::pair<float, int32_t> quant_output = quantizationParams<uint8_t>(kMinOut, kMax);
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_output.first, quant_output.second);

  std::vector<uint8_t> quantize_input{
      127, 67, 160, 192, //
      176, 85, 240, 255, //
  };
  input_tensor.writeData(quantize_input.data(), quantize_input.size() * sizeof(uint8_t));

  Relu kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray({0, 0, 66, 130, 98, 0, 226, 255}));
}

TEST(ReluTest, Input_Output_Type_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f});
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  Relu kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(ReluTest, Invalid_Input_Type_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::S64>({1}, {1});
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Relu kernel(&input_tensor, &output_tensor);
  kernel.configure();
  EXPECT_ANY_THROW(kernel.execute());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
