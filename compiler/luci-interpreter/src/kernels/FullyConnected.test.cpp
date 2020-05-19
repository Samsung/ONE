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

#include "kernels/FullyConnected.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(FullyConnectedTest, Float)
{
  Shape input_shape{3, 2, 2, 1};
  std::vector<float> input_data{
      -3, -5, 5,  4, 9,  -2, // batch = 0
      -3, -2, -4, 9, -8, 1,  // batch = 1
  };
  Shape weights_shape{3, 6};
  std::vector<float> weights_data{
      -3, -7, 4, -4, -6, 4,  // unit = 0
      3,  5,  2, 3,  -3, -8, // unit = 1
      -3, 7,  4, 9,  0,  -5, // unit = 2
  };
  Shape bias_shape{3};
  std::vector<float> bias_data{-1, -5, -8};

  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor weights_tensor = makeInputTensor<DataType::FLOAT32>(weights_shape, weights_data);
  Tensor bias_tensor = makeInputTensor<DataType::FLOAT32>(bias_shape, bias_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  FullyConnectedParams params{};
  params.activation = Activation::RELU;

  FullyConnected kernel(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{
      0,  0,  32, // batch = 0
      22, 11, 47, // batch = 1
  };
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
