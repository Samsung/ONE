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

#include "kernels/Add.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class AddTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

// for quantized Add, the error shouldn't exceed step
float GetTolerance(float min, float max)
{
  float kQuantizedStep = (max - min) / 255.0;
  return kQuantizedStep;
}

TEST_F(AddTest, Uint8)
{
  std::initializer_list<int32_t> base_shape = {2, 3, 1, 2};
  std::initializer_list<float> base_data = {-0.3f, 2.3f, 0.9f,  0.5f, 0.8f, -1.1f,
                                            1.2f,  2.8f, -1.6f, 0.0f, 0.7f, -2.2f};
  std::initializer_list<int32_t> test_shapes[] = {
    {1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  std::initializer_list<float> test_data = {0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f};
  std::initializer_list<int32_t> output_shapes[] = {
    {2, 3, 3, 2}, {2, 3, 1, 2}, {2, 3, 3, 2}, {2, 3, 1, 2}};
  std::vector<std::vector<float>> output_data = {
    {-0.1f, 2.6f,  -0.7f, 2.8f,  0.7f,  3.0f,  1.1f, 0.8f,  0.5f, 1.0f,  1.9f, 1.4f,
     1.0f,  -0.8f, 0.4f,  -0.6f, 1.8f,  -0.2f, 1.4f, 3.0f,  0.8f, 3.0f,  2.2f, 3.0f,
     -1.4f, 0.3f,  -2.0f, 0.5f,  -0.6f, 0.9f,  0.9f, -1.9f, 0.3f, -1.7f, 1.7f, -1.3f},
    {-0.1f, 2.6f, 0.5f, 1.0f, 1.8f, -0.2f, 1.4f, 3.0f, -2.0f, 0.5f, 1.7f, -1.3f},
    {-0.1f, 2.5f,  0.0f,  2.6f,  -0.7f, 1.9f,  1.1f, 0.7f,  1.2f, 0.8f,  0.5f, 0.1f,
     1.0f,  -0.9f, 1.1f,  -0.8f, 0.4f,  -1.5f, 1.7f, 3.0f,  2.2f, 3.0f,  2.1f, 3.0f,
     -1.1f, 0.5f,  -0.6f, 1.0f,  -0.7f, 0.9f,  1.2f, -1.7f, 1.7f, -1.2f, 1.6f, -1.3f},
    {-0.1f, 2.5f, 1.2f, 0.8f, 0.4f, -1.5f, 1.7f, 3.0f, -0.6f, 1.0f, 1.6f, -1.3f}};
  float kQuantizedTolerance = GetTolerance(-3.f, 3.f);
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-3.f, 3.f);
  for (int i = 0; i < output_data.size(); i++)
  {
    Tensor input1_tensor = makeInputTensor<DataType::U8>(
      base_shape, quant_param.first, quant_param.second, base_data, _memory_manager.get());
    Tensor input2_tensor = makeInputTensor<DataType::U8>(
      test_shapes[i], quant_param.first, quant_param.second, test_data, _memory_manager.get());
    Tensor output_tensor =
      makeOutputTensor(getElementType<uint8_t>(), quant_param.first, quant_param.second);

    AddParams params{};
    params.activation = Activation::NONE;

    Add kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
    kernel.configure();
    _memory_manager->allocate_memory(output_tensor);
    kernel.execute();

    EXPECT_THAT(dequantizeTensorData(output_tensor),
                FloatArrayNear(output_data[i], kQuantizedTolerance));
    EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shapes[i]));
  }
  // Re-run with exchanged inputs.
  for (int i = 0; i < output_data.size(); i++)
  {
    Tensor input1_tensor = makeInputTensor<DataType::U8>(
      test_shapes[i], quant_param.first, quant_param.second, test_data, _memory_manager.get());
    Tensor input2_tensor = makeInputTensor<DataType::U8>(
      base_shape, quant_param.first, quant_param.second, base_data, _memory_manager.get());
    Tensor output_tensor =
      makeOutputTensor(getElementType<uint8_t>(), quant_param.first, quant_param.second);

    AddParams params{};
    params.activation = Activation::NONE;

    Add kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
    kernel.configure();
    _memory_manager->allocate_memory(output_tensor);
    kernel.execute();

    EXPECT_THAT(dequantizeTensorData(output_tensor),
                FloatArrayNear(output_data[i], kQuantizedTolerance));
    EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shapes[i]));
  }
}

TEST_F(AddTest, Float)
{
  Shape base_shape = {2, 3, 1, 2};
  std::vector<Shape> test_shapes{{1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  std::vector<std::vector<float>> test_outputs = {
    {0.0f, 2.6f, 0.0f, 2.8f, 0.7f, 3.2f, 1.1f, 0.8f, 0.5f, 1.0f, 1.9f, 1.4f,
     1.0f, 0.0f, 0.4f, 0.0f, 1.8f, 0.0f, 1.4f, 3.1f, 0.8f, 3.3f, 2.2f, 3.7f,
     0.0f, 0.3f, 0.0f, 0.5f, 0.0f, 0.9f, 0.9f, 0.0f, 0.3f, 0.0f, 1.7f, 0.0f},
    {0.0f, 2.6f, 0.5f, 1.0f, 1.8f, 0.0f, 1.4f, 3.1f, 0.0f, 0.5f, 1.7f, 0.0f},
    {0.0f, 2.5f, 0.0f, 2.6f, 0.0f, 1.9f, 1.1f, 0.7f, 1.2f, 0.8f, 0.5f, 0.1f,
     1.0f, 0.0f, 1.1f, 0.0f, 0.4f, 0.0f, 1.7f, 3.3f, 2.2f, 3.8f, 2.1f, 3.7f,
     0.0f, 0.5f, 0.0f, 1.0f, 0.0f, 0.9f, 1.2f, 0.0f, 1.7f, 0.0f, 1.6f, 0.0f},
    {0.0f, 2.5f, 1.2f, 0.8f, 0.4f, 0.0f, 1.7f, 3.3f, 0.0f, 1.0f, 1.6f, 0.0f}};
  std::vector<float> input1_data{-0.3f, 2.3f, 0.9f,  0.5f, 0.8f, -1.1f,
                                 1.2f,  2.8f, -1.6f, 0.0f, 0.7f, -2.2f};
  std::vector<float> input2_data{0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f};
  for (size_t i = 0; i < test_shapes.size(); ++i)
  {
    Tensor input1_tensor =
      makeInputTensor<DataType::FLOAT32>(base_shape, input1_data, _memory_manager.get());
    Tensor input2_tensor =
      makeInputTensor<DataType::FLOAT32>(test_shapes[i], input2_data, _memory_manager.get());
    Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

    AddParams params{};
    params.activation = Activation::RELU;

    Add kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
    kernel.configure();
    _memory_manager->allocate_memory(output_tensor);
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(test_outputs[i], 0.0001f))
      << "With shape number " << i;
  }
  // Re-run with exchanged inputs.
  for (size_t i = 0; i < test_shapes.size(); ++i)
  {
    Tensor input1_tensor =
      makeInputTensor<DataType::FLOAT32>(test_shapes[i], input2_data, _memory_manager.get());
    Tensor input2_tensor =
      makeInputTensor<DataType::FLOAT32>(base_shape, input1_data, _memory_manager.get());
    Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

    AddParams params{};
    params.activation = Activation::RELU;

    Add kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
    kernel.configure();
    _memory_manager->allocate_memory(output_tensor);
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(test_outputs[i], 0.0001f))
      << "With shape number " << i;
  }
}

TEST_F(AddTest, SInt16)
{
  Shape base_shape = {2, 3, 1, 2};
  std::vector<Shape> test_shapes{{1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  std::vector<std::vector<int32_t>> ref_output_shapes{
    {2, 3, 3, 2}, {2, 3, 1, 2}, {2, 3, 3, 2}, {2, 3, 1, 2}};

  std::vector<float> input1_data{-0.3f, 2.3f, 0.9f,  0.5f, 0.8f, -1.1f,
                                 1.2f,  2.8f, -1.6f, 0.0f, 0.7f, -2.2f};
  std::vector<float> input2_data{0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f};
  std::vector<std::vector<float>> ref_outputs = {
    {0.0f, 2.6f, 0.0f, 2.8f, 0.7f, 3.2f, 1.1f, 0.8f, 0.5f, 1.0f, 1.9f, 1.4f,
     1.0f, 0.0f, 0.4f, 0.0f, 1.8f, 0.0f, 1.4f, 3.1f, 0.8f, 3.3f, 2.2f, 3.7f,
     0.0f, 0.3f, 0.0f, 0.5f, 0.0f, 0.9f, 0.9f, 0.0f, 0.3f, 0.0f, 1.7f, 0.0f},
    {0.0f, 2.6f, 0.5f, 1.0f, 1.8f, 0.0f, 1.4f, 3.1f, 0.0f, 0.5f, 1.7f, 0.0f},
    {0.0f, 2.5f, 0.0f, 2.6f, 0.0f, 1.9f, 1.1f, 0.7f, 1.2f, 0.8f, 0.5f, 0.1f,
     1.0f, 0.0f, 1.1f, 0.0f, 0.4f, 0.0f, 1.7f, 3.3f, 2.2f, 3.8f, 2.1f, 3.7f,
     0.0f, 0.5f, 0.0f, 1.0f, 0.0f, 0.9f, 1.2f, 0.0f, 1.7f, 0.0f, 1.6f, 0.0f},
    {0.0f, 2.5f, 1.2f, 0.8f, 0.4f, 0.0f, 1.7f, 3.3f, 0.0f, 1.0f, 1.6f, 0.0f}};

  for (size_t i = 0; i < test_shapes.size(); ++i)
  {
    Tensor input1_tensor = makeInputTensor<DataType::S16>(base_shape, 3.0 / 32767, 0, input1_data,
                                                          _memory_manager.get());
    Tensor input2_tensor = makeInputTensor<DataType::S16>(test_shapes[i], 1.0 / 32767, 0,
                                                          input2_data, _memory_manager.get());
    Tensor output_tensor = makeOutputTensor(DataType::S16, 4.0 / 32767, 0);
    const float tolerance = output_tensor.scale();

    AddParams params{};
    params.activation = Activation::RELU;

    Add kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
    kernel.configure();
    _memory_manager->allocate_memory(output_tensor);
    kernel.execute();

    EXPECT_THAT(extractTensorShape(output_tensor),
                ::testing::ElementsAreArray(ref_output_shapes[i]))
      << "With shape number " << i;
    EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_outputs[i], tolerance))
      << "With shape number " << i;
  }
  // Re-run with exchanged inputs and different scales.
  for (size_t i = 0; i < test_shapes.size(); ++i)
  {
    Tensor input1_tensor = makeInputTensor<DataType::S16>(test_shapes[i], 2.0 / 32767, 0,
                                                          input2_data, _memory_manager.get());
    Tensor input2_tensor = makeInputTensor<DataType::S16>(base_shape, 4.0 / 32767, 0, input1_data,
                                                          _memory_manager.get());
    Tensor output_tensor = makeOutputTensor(DataType::S16, 5.0 / 32767, 0);
    const float tolerance = output_tensor.scale();

    AddParams params{};
    params.activation = Activation::RELU;

    Add kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
    kernel.configure();
    _memory_manager->allocate_memory(output_tensor);
    kernel.execute();

    EXPECT_THAT(extractTensorShape(output_tensor),
                ::testing::ElementsAreArray(ref_output_shapes[i]))
      << "With shape number " << i;
    EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_outputs[i], tolerance))
      << "With shape number " << i;
  }
}

TEST_F(AddTest, Input_Output_Type_NEG)
{
  Tensor input1_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f}, _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::S32>({1}, {2}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  AddParams params{};
  params.activation = Activation::RELU;

  Add kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(AddTest, Invalid_Input_Type_NEG)
{
  Tensor input1_tensor = makeInputTensor<DataType::S64>({1}, {1}, _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::S64>({1}, {2}, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  AddParams params{};
  params.activation = Activation::RELU;

  Add kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  EXPECT_ANY_THROW(kernel.execute());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
