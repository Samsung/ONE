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
#include "kernels/MaxPool2D.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

class MaxPool2DTest : public ::testing::Test
{
protected:
  void SetUp() override { _memory_manager = std::make_unique<TestMemoryManager>(); }

  std::unique_ptr<IMemoryManager> _memory_manager;
};

TEST_F(MaxPool2DTest, Float)
{
  Shape input_shape{1, 3, 5, 1};
  std::vector<float> input_data{
    1,  -1, 0,  -2, 2,  //
    -7, -6, -5, -4, -3, //
    5,  4,  3,  6,  7,  //
  };
  Tensor input_tensor =
    makeInputTensor<DataType::FLOAT32>(input_shape, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 3;
  params.stride_height = 1;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  MaxPool2D kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{
    1, 2, //
    5, 6, //
  };
  std::initializer_list<int32_t> ref_output_shape{1, 2, 2, 1};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(MaxPool2DTest, Uint8)
{
  std::pair<float, int32_t> quant_param = quantizationParams<uint8_t>(-15.9375, 15.9375);
  std::vector<float> input_data{
    0,  -6, 12, 4, //
    -3, -2, 10, 7, //
  };
  Tensor input_tensor = makeInputTensor<DataType::U8>(
    {1, 2, 4, 1}, quant_param.first, quant_param.second, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8, quant_param.first, quant_param.second);

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 2;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  MaxPool2D kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  std::vector<float> ref_output_data{0.0, 6.0};
  std::initializer_list<int32_t> ref_output_shape{1, 1, 2, 1};
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
}

TEST_F(MaxPool2DTest, SInt16)
{
  Shape input_shape{1, 3, 5, 1};
  std::vector<int32_t> ref_output_shape{1, 2, 2, 1};
  std::vector<float> input_data{
    1,  -1, 0,  -2, 2,  //
    -7, -6, -5, -4, -3, //
    5,  4,  3,  6,  7,  //
  };
  std::vector<float> ref_output_data{
    1, 2, //
    5, 6, //
  };

  Tensor input_tensor =
    makeInputTensor<DataType::S16>(input_shape, 0.2, 0, input_data, _memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16, 0.2, 0);

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.filter_height = 2;
  params.filter_width = 3;
  params.stride_height = 1;
  params.stride_width = 2;
  params.activation = Activation::RELU6;

  MaxPool2D kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  _memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(ref_output_shape));
  EXPECT_THAT(dequantizeTensorData(output_tensor), FloatArrayNear(ref_output_data));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
#ednif
