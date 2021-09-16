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

#include "kernels/Mul.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class MulTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(MulTest, Float)
{
  Shape base_shape = {2, 3, 1, 2};
  std::vector<Shape> test_shapes{{1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  std::vector<std::vector<float>> test_outputs = {
    {0.00f, 0.69f, 0.12f, 1.15f, 0.00f, 2.07f, 0.18f, 0.15f, 0.00f, 0.25f, 0.90f, 0.45f,
     0.16f, 0.00f, 0.00f, 0.00f, 0.80f, 0.00f, 0.24f, 0.84f, 0.00f, 1.40f, 1.20f, 2.52f,
     0.00f, 0.00f, 0.64f, 0.00f, 0.00f, 0.00f, 0.14f, 0.00f, 0.00f, 0.00f, 0.70f, 0.00f},
    {0.00f, 0.69f, 0.00f, 0.25f, 0.80f, 0.00f, 0.24f, 0.84f, 0.64f, 0.00f, 0.70f, 0.00f},
    {0.00f, 0.46f, 0.00f, 0.69f, 0.12f, 0.00f, 0.18f, 0.10f, 0.27f, 0.15f, 0.00f, 0.00f,
     0.16f, 0.00f, 0.24f, 0.00f, 0.00f, 0.44f, 0.60f, 1.40f, 1.20f, 2.80f, 1.08f, 2.52f,
     0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.35f, 0.00f, 0.70f, 0.00f, 0.63f, 0.00f},
    {0.00f, 0.46f, 0.27f, 0.15f, 0.00f, 0.44f, 0.60f, 1.40f, 0.00f, 0.00f, 0.63f, 0.00f}};
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

    MulParams params{};
    params.activation = Activation::RELU;

    Mul kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
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

    MulParams params{};
    params.activation = Activation::RELU;

    Mul kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
    kernel.configure();
    _memory_manager->allocate_memory(output_tensor);
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(test_outputs[i], 0.0001f))
      << "With shape number " << i;
  }
}

TEST_F(MulTest, SInt16)
{
  Shape base_shape = {2, 3, 1, 2};
  std::vector<Shape> test_shapes{{1, 1, 3, 2}, {1, 3, 1, 2}, {2, 1, 3, 1}, {2, 3, 1, 1}};
  std::vector<std::vector<int32_t>> ref_output_shapes{
    {2, 3, 3, 2}, {2, 3, 1, 2}, {2, 3, 3, 2}, {2, 3, 1, 2}};

  std::vector<float> input1_data{-0.3f, 2.3f, 0.9f,  0.5f, 0.8f, -1.1f,
                                 1.2f,  2.8f, -1.6f, 0.0f, 0.7f, -2.2f};
  std::vector<float> input2_data{0.2f, 0.3f, -0.4f, 0.5f, 1.0f, 0.9f};
  std::vector<std::vector<float>> ref_outputs = {
    {0.00f, 0.69f, 0.12f, 1.15f, 0.00f, 2.07f, 0.18f, 0.15f, 0.00f, 0.25f, 0.90f, 0.45f,
     0.16f, 0.00f, 0.00f, 0.00f, 0.80f, 0.00f, 0.24f, 0.84f, 0.00f, 1.40f, 1.20f, 2.52f,
     0.00f, 0.00f, 0.64f, 0.00f, 0.00f, 0.00f, 0.14f, 0.00f, 0.00f, 0.00f, 0.70f, 0.00f},
    {0.00f, 0.69f, 0.00f, 0.25f, 0.80f, 0.00f, 0.24f, 0.84f, 0.64f, 0.00f, 0.70f, 0.00f},
    {0.00f, 0.46f, 0.00f, 0.69f, 0.12f, 0.00f, 0.18f, 0.10f, 0.27f, 0.15f, 0.00f, 0.00f,
     0.16f, 0.00f, 0.24f, 0.00f, 0.00f, 0.44f, 0.60f, 1.40f, 1.20f, 2.80f, 1.08f, 2.52f,
     0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.35f, 0.00f, 0.70f, 0.00f, 0.63f, 0.00f},
    {0.00f, 0.46f, 0.27f, 0.15f, 0.00f, 0.44f, 0.60f, 1.40f, 0.00f, 0.00f, 0.63f, 0.00f}};
  for (size_t i = 0; i < test_shapes.size(); ++i)
  {
    Tensor input1_tensor = makeInputTensor<DataType::S16>(base_shape, 3.0 / 32767, 0, input1_data,
                                                          _memory_manager.get());
    Tensor input2_tensor = makeInputTensor<DataType::S16>(test_shapes[i], 1.0 / 32767, 0,
                                                          input2_data, _memory_manager.get());
    Tensor output_tensor = makeOutputTensor(DataType::S16, 4.0 / 32767, 0);
    const float tolerance = output_tensor.scale() * 2;

    MulParams params{};
    params.activation = Activation::RELU;

    Mul kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
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
    Tensor output_tensor = makeOutputTensor(DataType::S16, 3.0 / 32767, 0);
    const float tolerance = output_tensor.scale() * 2;

    MulParams params{};
    params.activation = Activation::RELU;

    Mul kernel(&input1_tensor, &input2_tensor, &output_tensor, params);
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

} // namespace
} // namespace kernels
} // namespace luci_interpreter
