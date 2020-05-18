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

#include "kernels/Softmax.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

TEST(SoftmaxTest, Float)
{
  Shape input_shape{2, 1, 2, 3};
  std::vector<float> input_data{
      5,  -9, 8,  //
      -7, 2,  -4, //
      1,  -2, 9,  //
      3,  -6, -1, //
  };
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  SoftmaxParams params{};
  params.beta = 0.1;

  Softmax kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  std::vector<float> ref_output_data{
      0.38514, 0.09497, 0.51989, //
      0.20792, 0.51141, 0.28067, //
      0.25212, 0.18678, 0.56110, //
      0.48149, 0.19576, 0.32275, //
  };
  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ElementsAreArray(ArrayFloatNear(ref_output_data)));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
