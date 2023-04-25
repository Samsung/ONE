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

#include "kernels/AveragePool2D.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class AveragePool2DTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(AveragePool2DTest, Float)
{
  Shape input_shape{1, 3, 5, 1};
  std::vector<float> input_data{
    -4, -3, -2, -1, 0,  //
    1,  2,  3,  4,  5,  //
    6,  7,  8,  9,  10, //
  };
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor scratchpad(DataType::FLOAT32, Shape({}), {}, "");

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 3;
  params.stride_height = 1;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  AveragePool2D kernel(&input_tensor, &output_tensor, &scratchpad, params);
  kernel.configure();
  _memory_manager->allocate_memory(scratchpad);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{
    0, 1.5, //
    4.5, 6, //
  };
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 2, 2, 1}));
}

TEST_F(AveragePool2DTest, Uint8_0)
{
  std::vector<float> input_data{
    0,  -6, 12, 4, //
    -3, -2, 10, 7, //
  };
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-15.9375f, 15.9375f);
  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 4, 1}, quant_param.first, quant_param.second, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);
  Tensor scratchpad(DataType::U8, Shape({}), {}, "");

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 2;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  AveragePool2D kernel(&input_tensor, &output_tensor, &scratchpad, params);
  kernel.configure();
  _memory_manager->allocate_memory(scratchpad);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear({0.0, 6.0}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 1, 2, 1}));
}

TEST_F(AveragePool2DTest, Uint8_1)
{
  std::vector<float> input_data{
    0, 6, 12, 4, //
    3, 2, 10, 7, //
  };

  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-15.9375f, 15.9375f);
  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 4, 1}, quant_param.first, quant_param.second, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);
  Tensor scratchpad(DataType::U8, Shape({}), {}, "");

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 2;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  AveragePool2D kernel(&input_tensor, &output_tensor, &scratchpad, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  _memory_manager->allocate_memory(scratchpad);
  kernel.execute();

  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear({2.75, 6.0}));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray({1, 1, 2, 1}));
}

TEST_F(AveragePool2DTest, SInt16)
{
  Shape input_shape{1, 3, 5, 1};
  std::vector<int32_t> ref_output_shape{1, 2, 2, 1};
  std::vector<float> input_data{
    -4, -3, -2, -1, 0,  //
    1,  2,  3,  4,  5,  //
    6,  7,  8,  9,  10, //
  };
  std::vector<float> ref_output_data{
    0, 1.5, //
    4.5, 6, //
  };
  Tensor input_tensor =
    makeInputTensor<DataType::S16>(input_shape, 0.5, 0, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.5, 0);
  Tensor scratchpad(DataType::S16, Shape({}), {}, "");

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 3;
  params.stride_height = 1;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  AveragePool2D kernel(&input_tensor, &output_tensor, &scratchpad, params);
  kernel.configure();
  _memory_manager->allocate_memory(scratchpad);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST_F(AveragePool2DTest, SInt8)
{
  Shape input_shape{1, 4, 5, 1};
  std::vector<int32_t> ref_output_shape{1, 2, 2, 1};
  std::vector<float> input_data{-7, -3, 0,  2, -5, 12, -15, 3,  10, 5,
                                7,  -6, -1, 9, -2, 0,  -5,  11, -1, -7};
  std::vector<float> ref_output_data{
    0, 2.5, //
    1, 1.5, //
  };

  std::pair<float, int32_t> quant_param = quantizationParams<int8_t>(-15.9375f, 15.9375f);
  Tensor input_tensor = makeInputTensor<DataType::S8>(
    input_shape, quant_param.first, quant_param.second, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S8, quant_param.first, quant_param.second);
  Tensor scratchpad(DataType::S8, Shape({}), {}, "");

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 3;
  params.stride_height = 2;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  AveragePool2D kernel(&input_tensor, &output_tensor, &scratchpad, params);
  kernel.configure();
  _memory_manager->allocate_memory(scratchpad);
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

TEST_F(AveragePool2DTest, Invalid_Input_Shape_NEG)
{
  Shape input_shape{1, 3, 5};
  std::vector<float> input_data{
    -4, -3, -2, -1, 0,  //
    1,  2,  3,  4,  5,  //
    6,  7,  8,  9,  10, //
  };
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);
  Tensor scratchpad(DataType::FLOAT32, Shape({}), {}, "");

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 3;
  params.stride_height = 1;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  AveragePool2D kernel(&input_tensor, &output_tensor, &scratchpad, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(AveragePool2DTest, In_Out_Type_NEG)
{
  Shape input_shape{1, 3, 5, 1};
  std::vector<float> input_data{
    -4, -3, -2, -1, 0,  //
    1,  2,  3,  4,  5,  //
    6,  7,  8,  9,  10, //
  };
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8);
  Tensor scratchpad(DataType::FLOAT32, Shape({}), {}, "");

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 3;
  params.stride_height = 1;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  AveragePool2D kernel(&input_tensor, &output_tensor, &scratchpad, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST_F(AveragePool2DTest, Quant_Param_NEG)
{
  std::vector<float> input_data{
    0,  -6, 12, 4, //
    -3, -2, 10, 7, //
  };

  std::pair<float, int32_t> quant_param1 = quantizationParams<uint8_t>(-15.9375f, 15.9375f);
  std::pair<float, int32_t> quant_param2 = quantizationParams<uint8_t>(-7.875f, 7.875f);
  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 4, 1}, quant_param1.first, quant_param1.second, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param2.first, quant_param2.second);
  Tensor scratchpad(DataType::U8, Shape({}), {}, "");

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 2;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  AveragePool2D kernel(&input_tensor, &output_tensor, &scratchpad, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
