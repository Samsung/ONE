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

#include "kernels/Relu6.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(Relu6Test, FloatSimple)
{
  std::vector<float> input_data{
      0.0f, 1.0f,  3.0f,  // Row 1
      7.0f, -1.0f, -2.0f, // Row 2
  };

  std::vector<float> ref_output_data{
      0.0f, 1.0f, 3.0f, // Row 1
      6.0f, 0.0f, 0.0f, // Row 2
  };

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({2, 3}, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Relu6 kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({2, 3}));
}

TEST(Relu6Test, Uint8Quantized)
{
  // Choose min / max in such a way that there are exactly 256 units to avoid rounding errors.
  const float f_min = (-128.0 / 128.0) * 10;
  const float f_max = (127.0 / 128.0) * 10;
  const float tolerance = (f_max - f_min) / 255.0;

  std::vector<float> input_data{
      0,  -6, 2, 8, //
      -2, 3,  7, 1, //
  };

  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(f_min, f_max);
  Tensor input_tensor = makeInputTensor<DataType::U8>({1, 2, 4, 1}, quant_param.first,
                                                      quant_param.second, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  Relu6 kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray({128, 128, 154, 205, 128, 166, 205, 141}));
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear({0, 0, 2, 6, 0, 3, 6, 1}, tolerance));
}

TEST(Relu6Test, Uint8Requantized)
{
  // Choose min / max in such a way that there are exactly 256 units to avoid rounding errors.
  const float in_min = (-128.0 / 128.0) * 10;
  const float in_max = (127.0 / 128.0) * 10;
  const float out_min = (0.0 / 256.0) * 0;
  const float out_max = (255.0 / 256.0) * 6;
  const float tolerance = (in_max - in_min) / 255.0;

  std::vector<float> input_data{
      0,  -6, 2, 8, //
      -2, 3,  7, 1, //
  };

  std::pair<float, int32_t> quant_input = quantizationParams<uint8_t>(in_min, in_max);
  Tensor input_tensor = makeInputTensor<DataType::U8>({1, 2, 4, 1}, quant_input.first,
                                                      quant_input.second, input_data);

  std::pair<float, int32_t> quant_output = quantizationParams<uint8_t>(out_min, out_max);
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_output.first, quant_output.second);

  Relu6 kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 4, 1}));
  EXPECT_THAT(extractTensorData<uint8_t>(output_tensor),
              ::testing::ElementsAreArray({0, 0, 87, 255, 0, 127, 255, 43}));
  EXPECT_THAT(dequantizeTensorData(output_tensor),
              FloatArrayNear({0, 0, 2, 6, 0, 3, 6, 1}, tolerance));
}

TEST(Relu6Test, Input_Output_Type_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f});
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  Relu6 kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(Relu6Test, Invalid_Input_Type_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::S64>({1}, {1});
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Relu6 kernel(&input_tensor, &output_tensor);
  kernel.configure();
  EXPECT_ANY_THROW(kernel.execute());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
