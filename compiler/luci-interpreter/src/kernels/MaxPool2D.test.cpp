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

#include "kernels/MaxPool2D.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(MaxPool2DTest, Float)
{
  Shape input_shape{1, 3, 5, 1};
  std::vector<float> input_data{
      1,  -1, 0,  -2, 2,  //
      -7, -6, -5, -4, -3, //
      5,  4,  3,  6,  7,  //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
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
  kernel.execute();

  std::vector<float> ref_output_data{
      1, 2, //
      5, 6, //
  };
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
