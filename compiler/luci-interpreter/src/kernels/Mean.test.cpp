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

#include "kernels/Mean.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(MeanTest, FloatKeepDims)
{
  std::vector<float> input_data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                   9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                   17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

  std::vector<int32_t> axis_data{0, 2};
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({4, 3, 2}, input_data);
  Tensor axis_tensor = makeInputTensor<DataType::S32>({2}, axis_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ReducerParams params{};
  params.keep_dims = true;

  Mean kernel(&input_tensor, &axis_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{10.5, 12.5, 14.5};
  std::initializer_list<int32_t> ref_output_shape{1, 3, 1};
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(MeanTest, FloatKeepDims4DMean)
{
  std::vector<float> input_data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                   9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                   17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

  std::vector<int32_t> axis_data{1, 2};
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({2, 2, 3, 2}, input_data);
  Tensor axis_tensor = makeInputTensor<DataType::S32>({2}, axis_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ReducerParams params{};
  params.keep_dims = true;

  Mean kernel(&input_tensor, &axis_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{6, 7, 18, 19};
  std::initializer_list<int32_t> ref_output_shape{2, 1, 1, 2};
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(MeanTest, FloatNotKeepDims)
{
  std::vector<float> input_data = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                                   9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                                   17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

  std::vector<int32_t> axis_data{1, 0, -3, -3};
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({4, 3, 2}, input_data);
  Tensor axis_tensor = makeInputTensor<DataType::S32>({4}, axis_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  ReducerParams params{};
  params.keep_dims = false;

  Mean kernel(&input_tensor, &axis_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{12, 13};
  std::initializer_list<int32_t> ref_output_shape{2};
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(MeanTest, Uint8KeepDims)
{
  float kQuantizedTolerance = getTolerance(-1.0, 1.0, 255);
  std::vector<float> input_data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-1.0f, 1.0f);

  std::vector<int32_t> axis_data{1};
  Tensor input_tensor{DataType::U8, {3, 2}, {{quant_param.first}, {quant_param.second}}, ""};
  Tensor axis_tensor = makeInputTensor<DataType::S32>({1}, axis_data);
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);
  std::vector<uint8_t> quantize_input =
      quantize<uint8_t>(input_data, quant_param.first, quant_param.second);
  input_tensor.writeData(quantize_input.data(), quantize_input.size() * sizeof(uint8_t));

  ReducerParams params{};
  params.keep_dims = true;

  Mean kernel(&input_tensor, &axis_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{0.3, 0.35, 0.55};
  std::initializer_list<int32_t> ref_output_shape{3, 1};
  EXPECT_THAT(dequantize<uint8_t>(extractTensorData<uint8_t>(output_tensor), output_tensor.scale(),
                                  output_tensor.zero_point()),
              ElementsAreArray(ArrayFloatNear(ref_output_data, kQuantizedTolerance)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST(MeanTest, Uint8NotKeepDims)
{
  float kQuantizedTolerance = getTolerance(-1.0, 1.0, 255);
  std::vector<float> input_data = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-1.0f, 1.0f);

  std::vector<int32_t> axis_data{1};
  Tensor input_tensor{DataType::U8, {1, 3, 2}, {{quant_param.first}, {quant_param.second}}, ""};
  Tensor axis_tensor = makeInputTensor<DataType::S32>({1}, axis_data);
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);
  std::vector<uint8_t> quantize_input =
      quantize<uint8_t>(input_data, quant_param.first, quant_param.second);
  input_tensor.writeData(quantize_input.data(), quantize_input.size() * sizeof(uint8_t));

  ReducerParams params{};
  params.keep_dims = false;

  Mean kernel(&input_tensor, &axis_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{0.4, 0.4};
  std::initializer_list<int32_t> ref_output_shape{1, 2};
  EXPECT_THAT(dequantize<uint8_t>(extractTensorData<uint8_t>(output_tensor), output_tensor.scale(),
                                  output_tensor.zero_point()),
              ElementsAreArray(ArrayFloatNear(ref_output_data, kQuantizedTolerance)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
