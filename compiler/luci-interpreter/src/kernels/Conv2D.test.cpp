/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Conv2D.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(Conv2DTest, Float)
{
  Shape input_shape{1, 4, 3, 2};
  Shape filter_shape{2, 2, 2, 2};
  Shape bias_shape{2};
  std::vector<float> input_data{
      1,  2,  3,  4,  5,  6,  // row = 0
      7,  8,  9,  10, 11, 12, // row = 1
      13, 14, 15, 16, 17, 18, // row = 2
      19, 20, 21, 22, 23, 24, // row = 3
  };
  std::vector<float> filter_data{
      1,  2,  -3, -4, // out = 0, row = 0
      -5, 6,  -7, 8,  // out = 1, row = 0
      4,  -2, 3,  -1, // out = 0, row = 1
      -8, -6, 7,  5,  // out = 1, row = 1
  };
  std::vector<float> bias_data{1, 2};
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor filter_tensor = makeInputTensor<DataType::FLOAT32>(filter_shape, filter_data);
  Tensor bias_tensor = makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Conv2DParams params{};
  params.padding = Padding::VALID;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.activation = Activation::RELU;

  Conv2D kernel(&input_tensor, &filter_tensor, &bias_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{
      11, 16, 7, 20, // row = 0
      0,  40, 0, 44, // row = 1
  };
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
