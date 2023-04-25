/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
// TODO enable it
#if 0
#include "kernels/Concatenation.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class ConcatenationTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(ConcatenationTest, Float)
{
  std::vector<float> input1_data{1, 2, 3, 4, 5, 6};
  std::vector<float> input2_data{7, 8, 9, 10, 11, 12};
  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3}, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3}, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  ConcatenationParams params{};

  // Try different 'axis' and expect different results.
  {
    params.axis = 0;
    params.activation = luci::FusedActFunc::NONE;

    Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
    kernel.configure();
    for (auto t : kernel.getOutputTensors())
    {
      _memory_manager->allocate_memory(*t);
    }
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor),
                FloatArrayNear({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
  }
  {
    params.axis = -2; // Same as '0'.
    params.activation = luci::FusedActFunc::NONE;

    Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
    kernel.configure();
    _memory_manager->allocate_memory(output_tensor);
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor),
                FloatArrayNear({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
  }
  {
    params.axis = 1;
    params.activation = luci::FusedActFunc::NONE;

    Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
    kernel.configure();
    _memory_manager->allocate_memory(output_tensor);
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor),
                FloatArrayNear({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
  }
  {
    params.axis = -1; // Same as '1'.
    params.activation = luci::FusedActFunc::NONE;

    Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
    kernel.configure();
    _memory_manager->allocate_memory(output_tensor);
    kernel.execute();

    EXPECT_THAT(extractTensorData<float>(output_tensor),
                FloatArrayNear({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12}));
  }
}

TEST_F(ConcatenationTest, Input_Number_Check_NEG)
{
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  ConcatenationParams params{};

  params.axis = -1;
  params.activation = luci::FusedActFunc::NONE;

  Concatenation kernel({}, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(ConcatenationTest, Invalid_Axis_NEG)
{
  std::vector<float> input1_data{1, 2, 3, 4, 5, 6};
  std::vector<float> input2_data{7, 8, 9, 10, 11, 12};
  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3}, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3}, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  ConcatenationParams params{};

  params.axis = -3;
  params.activation = luci::FusedActFunc::NONE;

  Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(ConcatenationTest, Mismatching_Input_Type_NEG)
{
  std::vector<float> input1_data{1, 2, 3, 4, 5, 6};
  std::vector<uint8_t> input2_data{7, 8, 9, 10, 11, 12};
  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3}, input1_data, _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::U8>({2, 3}, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  ConcatenationParams params{};

  params.axis = -1;
  params.activation = luci::FusedActFunc::NONE;

  Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(ConcatenationTest, Mismatching_Input_Dimension_Num_NEG)
{
  std::vector<float> input1_data{1, 2, 3, 4, 5, 6};
  std::vector<float> input2_data{7, 8, 9, 10, 11, 12};
  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3}, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::FLOAT32>({1, 2, 3}, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  ConcatenationParams params{};

  params.axis = -1;
  params.activation = luci::FusedActFunc::NONE;

  Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(ConcatenationTest, Mismatching_Input_Dimension_NEG)
{
  std::vector<float> input1_data{1, 2, 3, 4, 5, 6};
  std::vector<float> input2_data{7, 8, 9, 10, 11, 12, 13, 14, 15};
  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3}, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::FLOAT32>({3, 3}, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  ConcatenationParams params{};

  params.axis = -1;
  params.activation = luci::FusedActFunc::NONE;

  Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(ConcatenationTest, Int8_Mismatching_Input_Type_NEG)
{
  std::vector<uint8_t> input1_data{1, 2, 3, 4};
  std::vector<int8_t> input2_data{5, 6, 7, 8};
  Tensor input1_tensor = makeInputTensor<DataType::U8>({2, 2}, input1_data, _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::S8>({2, 2}, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S8);
  ConcatenationParams params{};

  params.axis = -1;
  params.activation = luci::FusedActFunc::NONE;

  Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(ConcatenationTest, Int8_Mismatching_Input_Output_Quant_Params_NEG)
{
  std::vector<float> input1_data{1, 2, 3, 4, 5, 6};
  std::vector<float> input2_data{7, 8, 9, 10, 11, 12};
  int quantized_dimension = 3;
  std::vector<float> scales{0.1, 0.2, 0.3};
  std::vector<int32_t> zero_points{1, -1, 1};

  Tensor input1_tensor = makeInputTensor<DataType::S8>(
    {1, 1, 2, 3}, scales, zero_points, quantized_dimension, input1_data, _memory_manager.get());
  Tensor input2_tensor = makeInputTensor<DataType::S8>(
    {1, 1, 2, 3}, scales, zero_points, quantized_dimension, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S8, scales.at(0), zero_points.at(0));
  ConcatenationParams params{};

  params.axis = -1;
  params.activation = luci::FusedActFunc::NONE;

  Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(ConcatenationTest, Int8_Mismatching_Zero_Point_NEG)
{
  std::vector<float> input1_data{1, 2, 3, 4};
  std::vector<float> input2_data{5, 6, 7, 8};
  float scale = 0.1;
  int32_t zero_point_1 = 1;
  int32_t zero_point_2 = -1;

  Tensor input1_tensor =
    makeInputTensor<DataType::S8>({2, 2}, scale, zero_point_1, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::S8>({2, 2}, scale, zero_point_2, input2_data, _memory_manager.get());

  Tensor output_tensor = makeOutputTensor(DataType::S8, scale, zero_point_1);
  ConcatenationParams params{};

  params.axis = -1;
  params.activation = luci::FusedActFunc::NONE;

  Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

// TODO: Remove this test when concat w/ fused_activation is supported
TEST_F(ConcatenationTest, With_Fused_Activation_NEG)
{
  std::vector<float> input1_data{1, 2, 3, 4, 5, 6};
  std::vector<float> input2_data{7, 8, 9, 10, 11, 12};
  Tensor input1_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3}, input1_data, _memory_manager.get());
  Tensor input2_tensor =
    makeInputTensor<DataType::FLOAT32>({2, 3}, input2_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  ConcatenationParams params{};

  params.axis = 1;
  params.activation = luci::FusedActFunc::RELU;

  Concatenation kernel({&input1_tensor, &input2_tensor}, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
#endif
