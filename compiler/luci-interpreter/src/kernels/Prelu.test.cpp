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

#include "kernels/Prelu.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> alpha_shape,
           std::initializer_list<int32_t> output_shape, std::initializer_list<T> input_data,
           std::initializer_list<T> alpha_data, std::initializer_list<T> output_data,
           DataType element_type)
{
  Tensor input_tensor{element_type, input_shape, {}, ""};
  input_tensor.writeData(input_data.begin(), input_data.size() * sizeof(T));

  Tensor alpha_tensor{element_type, alpha_shape, {}, ""};
  alpha_tensor.writeData(alpha_data.begin(), alpha_data.size() * sizeof(T));

  Tensor output_tensor = makeOutputTensor(element_type);

  Prelu kernel(&input_tensor, &alpha_tensor, &output_tensor);

  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

TEST(PreluTest, FloatSimple)
{
  Check<float>(/*input_shape=*/{2, 3}, /*alpha_shape=*/{2, 3},
               /*output_shape=*/{2, 3}, /*input_data=*/
               {
                   0.0f, 1.0f, 3.0f,   // Row 1
                   1.0f, -1.0f, -2.0f, // Row 2
               },
               /*alpha_data=*/
               {
                   0.0f, 0.5f, 0.1f, // Row 1
                   0.0f, 0.5f, 0.1f, // Row 2
               },
               /*output_data=*/
               {
                   0.0f, 1.0f, 3.0f,   // Row 1
                   1.0f, -0.5f, -0.2f, // Row 2
               },
               getElementType<float>());

  SUCCEED();
}

TEST(PreluTest, FloatBroadcast)
{
  Check<float>(/*input_shape=*/{1, 2, 2, 3}, /*alpha_shape=*/{1, 1, 3},
               /*output_shape=*/{1, 2, 2, 3}, /*input_data=*/
               {
                   0.0f, 0.0f, 0.0f,    // Row 1, Column 1
                   1.0f, 1.0f, 1.0f,    // Row 1, Column 2
                   -1.0f, -1.0f, -1.0f, // Row 2, Column 1
                   -2.0f, -2.0f, -2.0f, // Row 2, Column 2
               },
               /*alpha_data=*/
               {0.0f, 1.0f, 2.0f},
               /*output_data=*/
               {
                   0.0f, 0.0f, 0.0f,   // Row 1, Column 1
                   1.0f, 1.0f, 1.0f,   // Row 1, Column 2
                   0.0f, -1.0f, -2.0f, // Row 2, Column 1
                   0.0f, -2.0f, -4.0f, // Row 2, Column 2
               },
               getElementType<float>());

  SUCCEED();
}

float GetTolerance(float min, float max) { return (max - min) / 255.0; }

TEST(PreluTest, Uint8Simple)
{
  std::vector<float> input_data{-0.8f, 0.2f, 0.9f, 0.7f, 0.1f, -0.4f};
  std::vector<float> alpha_data{0.5f, 0.5f, 0.5f, 0.25f, 1.0f, 0.25f};
  std::vector<float> ref_output_data{-0.4f, 0.2f, 0.9f, 0.7f, 0.1f, -0.1f};

  float kQuantizedTolerance = GetTolerance(-1.0, 1.0);
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-1.0f, 1.0f);

  Tensor input_tensor{DataType::U8, {1, 2, 3, 1}, {{quant_param.first}, {quant_param.second}}, ""};
  Tensor alpha_tensor{DataType::U8, {1, 2, 3, 1}, {{quant_param.first}, {quant_param.second}}, ""};
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  std::vector<uint8_t> quantize_input =
      quantize<uint8_t>(input_data, quant_param.first, quant_param.second);
  input_tensor.writeData(quantize_input.data(), quantize_input.size() * sizeof(uint8_t));

  std::vector<uint8_t> quantize_alpha =
      quantize<uint8_t>(alpha_data, quant_param.first, quant_param.second);
  alpha_tensor.writeData(quantize_alpha.data(), quantize_alpha.size() * sizeof(uint8_t));

  Prelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(dequantize(extractTensorData<uint8_t>(output_tensor), output_tensor.scale(),
                         output_tensor.zero_point()),
              ElementsAreArray(ArrayFloatNear(ref_output_data, kQuantizedTolerance)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 3, 1}));

  SUCCEED();
}

TEST(PreluTest, Uint8Broadcast)
{
  std::vector<float> input_data{
      0.0f,   0.0f,   0.0f,   // Row 1, Column 1
      0.5f,   0.5f,   0.5f,   // Row 1, Column 2
      -1.0f,  -1.0f,  -1.0f,  // Row 2, Column 1
      -0.25f, -0.25f, -0.25f, // Row 2, Column 2
  };
  std::vector<float> alpha_data{0.0f, 0.5f, -0.5f};
  std::vector<float> ref_output_data{
      0.0f, 0.0f,    0.0f,  // Row 1, Column 1
      0.5f, 0.5f,    0.5f,  // Row 1, Column 2
      0.0f, -0.5f,   0.5f,  // Row 2, Column 1
      0.0f, -0.125f, 0.125f // Row 2, Column 2
  };
  std::vector<float> ref_quant_output_data{
      128, 128, 128, // Row 1, Column 1
      192, 192, 192, // Row 1, Column 2
      128, 64,  192, // Row 2, Column 1
      128, 112, 144  // Row 2, Column 2
  };
  float kQuantizedTolerance = 2 * (1. / 256);
  const float kMin = -1;
  const float kMax = 127.f / 128.f;
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(kMin, kMax);

  Tensor input_tensor{DataType::U8, {1, 2, 2, 3}, {{quant_param.first}, {quant_param.second}}, ""};
  Tensor alpha_tensor{DataType::U8, {1, 1, 3}, {{quant_param.first}, {quant_param.second}}, ""};
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  std::vector<uint8_t> quantize_input =
      quantize<uint8_t>(input_data, quant_param.first, quant_param.second);
  input_tensor.writeData(quantize_input.data(), quantize_input.size() * sizeof(uint8_t));

  std::vector<uint8_t> quantize_alpha =
      quantize<uint8_t>(alpha_data, quant_param.first, quant_param.second);
  alpha_tensor.writeData(quantize_alpha.data(), quantize_alpha.size() * sizeof(uint8_t));

  Prelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(dequantize(extractTensorData<uint8_t>(output_tensor), output_tensor.scale(),
                         output_tensor.zero_point()),
              ElementsAreArray(ArrayFloatNear(ref_output_data, kQuantizedTolerance)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 2, 3}));
  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray(ref_quant_output_data));
}

TEST(PreluTest, Input_Output_Type_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f});
  Tensor alpha_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f});
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  Prelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(PreluTest, Input_Alpha_Type_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f});
  Tensor alpha_tensor = makeInputTensor<DataType::U8>({1}, {1});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Prelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(PreluTest, Invalid_Input_Type_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::S64>({1}, {1});
  Tensor alpha_tensor = makeInputTensor<DataType::S64>({1}, {1});
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Prelu kernel(&input_tensor, &alpha_tensor, &output_tensor);
  kernel.configure();
  EXPECT_ANY_THROW(kernel.execute());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
