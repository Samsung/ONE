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

#include "kernels/Sub.h"
#include "kernels/TestUtils.h"

#include <algorithm>

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;
using std::pair;
using std::vector;
using std::transform;
using std::initializer_list;

// for quantized Add, the error shouldn't exceed step
float GetTolerance(float min, float max)
{
  float kQuantizedStep = (max - min) / 255.0;
  return kQuantizedStep;
}

TEST(SubTest, Uint8)
{
  Shape base_shape = {2, 3, 1, 2};
  vector<float> base_data = {-0.3f, 2.3f, 0.9f,  0.5f, 0.8f, -1.1f,
                             1.2f,  2.8f, -1.6f, 0.0f, 0.7f, -2.2f};
  vector<Shape> test_shapes = {{1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  vector<float> test_data = {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f};
  vector<vector<int32_t>> output_shapes = {{2, 3, 3, 2}, {2, 3, 1, 2}, {2, 3, 3, 2}, {2, 3, 1, 2}};
  vector<vector<float>> output_data = {
      {-0.5f, 2.0f,  0.1f,  1.8f,  -1.3f, 1.4f,  0.7f, 0.2f,  1.3f, 0.0f,  -0.1f, -0.4f,
       0.6f,  -1.4f, 1.2f,  -1.6f, -0.2f, -2.0f, 1.0f, 2.5f,  1.6f, 2.3f,  0.2f,  1.9f,
       -1.8f, -0.3f, -1.2f, -0.5f, -2.6f, -0.9f, 0.5f, -2.5f, 1.1f, -2.7f, -0.3f, -3.0f},
      {-0.5f, 2.0f, 1.3f, 0.0f, -0.2f, -2.0f, 1.0f, 2.5f, -1.2f, -0.5f, -0.3f, -3.0f},
      {-0.5f, 2.1f,  -0.6f, 2.0f,  0.1f,  2.7f,  0.7f, 0.3f,  0.6f,  0.2f,  1.3f,  0.9f,
       0.6f,  -1.3f, 0.5f,  -1.4f, 1.2f,  -0.7f, 0.7f, 2.3f,  0.2f,  1.8f,  0.3f,  1.9f,
       -2.1f, -0.5f, -2.6f, -1.0f, -2.5f, -0.9f, 0.2f, -2.7f, -0.3f, -3.0f, -0.2f, -3.0f},
      {-0.5f, 2.1f, 0.6f, 0.2f, 1.2f, -0.7f, 0.7f, 2.3f, -2.6f, -1.0f, -0.2f, -3.0f}};

  float kQuantizedTolerance = GetTolerance(-3.f, 3.f);
  pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-3.f, 3.f);
  for (size_t i = 0; i < output_data.size(); ++i)
  {
    Tensor input1_tensor =
        makeInputTensor<DataType::U8>(base_shape, quant_param.first, quant_param.second, base_data);
    Tensor input2_tensor = makeInputTensor<DataType::U8>(test_shapes[i], quant_param.first,
                                                         quant_param.second, test_data);
    Tensor output_tensor =
        makeOutputTensor(getElementType<uint8_t>(), quant_param.first, quant_param.second);

    SubParams params{};
    params.activation = Activation::NONE;

    Sub kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
    kernel.configure();
    kernel.execute();

    EXPECT_THAT(dequantizeTensorData(output_tensor),
                FloatArrayNear(output_data[i], kQuantizedTolerance));
    EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shapes[i]));
  }

  // Inversion step for output_data, because subtract is not commutative operation
  auto multiply = [](auto &i) {
    transform(i.begin(), i.end(), i.begin(), [](auto &value) { return value * -1.0f; });
  };
  for_each(output_data.begin(), output_data.end(), multiply);

  // Re-run with exchanged inputs.
  for (size_t i = 0; i < output_data.size(); ++i)
  {
    Tensor input1_tensor = makeInputTensor<DataType::U8>(test_shapes[i], quant_param.first,
                                                         quant_param.second, test_data);
    Tensor input2_tensor =
        makeInputTensor<DataType::U8>(base_shape, quant_param.first, quant_param.second, base_data);
    Tensor output_tensor =
        makeOutputTensor(getElementType<uint8_t>(), quant_param.first, quant_param.second);

    SubParams params{};
    params.activation = Activation::NONE;

    Sub kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
    kernel.configure();
    kernel.execute();

    EXPECT_THAT(dequantizeTensorData(output_tensor),
                FloatArrayNear(output_data[i], kQuantizedTolerance));
    EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shapes[i]));
  }
}

TEST(SubTest, Float)
{
  Shape base_shape = {2, 3, 1, 2};
  vector<Shape> test_shapes{{1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  vector<vector<int32_t>> output_shapes{{2, 3, 3, 2}, {2, 3, 1, 2}, {2, 3, 3, 2}, {2, 3, 1, 2}};
  vector<vector<float>> test_outputs = {
      {0.0f, 2.0f, 0.1f, 1.8f, 0.0f, 1.4f, 0.7f, 0.2f, 1.3f, 0.0f, 0.0f, 0.0f,
       0.6f, 0.0f, 1.2f, 0.0f, 0.0f, 0.0f, 1.0f, 2.5f, 1.6f, 2.3f, 0.2f, 1.9f,
       0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 1.1f, 0.0f, 0.0f, 0.0f},
      {0.0f, 2.0f, 1.3f, 0.0f, 0.0f, 0.0f, 1.0f, 2.5f, 0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 2.1f, 0.0f, 2.0f, 0.1f, 2.7f, 0.7f, 0.3f, 0.6f, 0.2f, 1.3f, 0.9f,
       0.6f, 0.0f, 0.5f, 0.0f, 1.2f, 0.0f, 0.7f, 2.3f, 0.2f, 1.8f, 0.3f, 1.9f,
       0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 2.1f, 0.6f, 0.2f, 1.2f, 0.0f, 0.7f, 2.3f, 0.0f, 0.0f, 0.0f, 0.0f}};

  vector<float> input1_data{-0.3f, 2.3f, 0.9f,  0.5f, 0.8f, -1.1f,
                            1.2f,  2.8f, -1.6f, 0.0f, 0.7f, -2.2f};
  vector<float> input2_data{0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f};
  for (size_t i = 0; i < test_shapes.size(); ++i)
  {
    Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>(base_shape, input1_data);
    Tensor input2_tensor = makeInputTensor<DataType::FLOAT32>(test_shapes[i], input2_data);
    Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

    SubParams params{};
    params.activation = Activation::RELU;

    Sub kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
    kernel.configure();
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(test_outputs[i], 0.0001f))
        << "With shape number " << i;

    EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shapes[i]));
  }
}

TEST(SubTest, Input_Output_Type_NEG)
{
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f});
  Tensor input2_tensor = makeInputTensor<DataType::S32>({1}, {2});
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  SubParams params{};
  params.activation = Activation::RELU;

  Sub kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(SubTest, Invalid_Input_Type_NEG)
{
  Tensor input1_tensor = makeInputTensor<DataType::S64>({1}, {1});
  Tensor input2_tensor = makeInputTensor<DataType::S64>({1}, {2});
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  SubParams params{};
  params.activation = Activation::RELU;

  Sub kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
  kernel.configure();
  EXPECT_ANY_THROW(kernel.execute());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
