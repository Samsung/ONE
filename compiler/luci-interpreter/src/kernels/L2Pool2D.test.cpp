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

#include "kernels/L2Pool2D.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(L2Pool2DTest, FloatNone)
{
  Shape input_shape{1, 2, 4, 1};
  std::vector<float> input_data{
    0, 6, 2,  4, //
    3, 2, 10, 7, //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.activation = Activation::NONE;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 2;
  params.stride_width = 2;

  L2Pool2D kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{3.5, 6.5};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  // TODO make a Shape checking of output_tensor.
}

TEST(L2Pool2DTest, FloatRelu)
{
  Shape input_shape{1, 2, 4, 1};
  std::vector<float> input_data{
    -1, -6, 2,  4, //
    -3, -2, 10, 7, //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.activation = Activation::RELU;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 2;
  params.stride_width = 2;

  L2Pool2D kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{3.53553, 6.5};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  // TODO make a Shape checking of output_tensor.
}

TEST(L2Pool2DTest, FloatRelu1)
{
  Shape input_shape{1, 2, 4, 1};
  std::vector<float> input_data{
    -0.1, -0.6, 2,  4, //
    -0.3, -0.2, 10, 7, //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.activation = Activation::RELU_N1_TO_1;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 2;
  params.stride_width = 2;

  L2Pool2D kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{0.353553, 1.0};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  // TODO make a Shape checking of output_tensor.
}

TEST(L2Pool2DTest, FloatRelu6)
{
  Shape input_shape{1, 2, 4, 1};
  std::vector<float> input_data{
    -0.1, -0.6, 2,  4, //
    -0.3, -0.2, 10, 7, //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.activation = Activation::RELU6;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 2;
  params.stride_width = 2;

  L2Pool2D kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{0.353553, 6.0};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  // TODO make a Shape checking of output_tensor.
}

TEST(L2Pool2DTest, FloatPaddingSame)
{
  Shape input_shape{1, 2, 4, 1};
  std::vector<float> input_data{
    0, 6, 2,  4, //
    3, 2, 10, 7, //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pool2DParams params{};
  params.padding = Padding::SAME;
  params.activation = Activation::NONE;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 2;
  params.stride_width = 2;

  L2Pool2D kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{3.5, 6.5};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  // TODO make a Shape checking of output_tensor.
}

TEST(L2Pool2DTest, FloatPaddingSameStride)
{
  Shape input_shape{1, 2, 4, 1};
  std::vector<float> input_data{
    0, 6, 2,  4, //
    3, 2, 10, 7, //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pool2DParams params{};
  params.padding = Padding::SAME;
  params.activation = Activation::NONE;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 1;
  params.stride_width = 1;

  L2Pool2D kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{3.5, 6.0, 6.5, 5.70088, 2.54951, 7.2111, 8.63134, 7.0};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  // TODO make a Shape checking of output_tensor.
}

TEST(L2Pool2DTest, FloatPaddingValidStride)
{
  Shape input_shape{1, 2, 4, 1};
  std::vector<float> input_data{
    0, 6, 2,  4, //
    3, 2, 10, 7, //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.activation = Activation::NONE;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 1;
  params.stride_width = 1;

  L2Pool2D kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{3.5, 6.0, 6.5};
  EXPECT_THAT(extractTensorData<float>(output_tensor), FloatArrayNear(ref_output_data));
  // TODO make a Shape checking of output_tensor.
}

TEST(L2Pool2DTest, InvalidInputShape_NEG)
{
  Shape input_shape{1, 2, 4};
  std::vector<float> input_data{
    0, 6, 2,  4, //
    3, 2, 10, 7, //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.activation = Activation::NONE;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 1;
  params.stride_width = 1;

  L2Pool2D kernel(&input_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(L2Pool2DTest, InvalidInputOutputType_NEG)
{
  Shape input_shape{1, 2, 4};
  std::vector<float> input_data{
    0, 6, 2,  4, //
    3, 2, 10, 7, //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  Pool2DParams params{};
  params.padding = Padding::VALID;
  params.activation = Activation::NONE;
  params.filter_height = 2;
  params.filter_width = 2;
  params.stride_height = 1;
  params.stride_width = 1;

  L2Pool2D kernel(&input_tensor, &output_tensor, params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
