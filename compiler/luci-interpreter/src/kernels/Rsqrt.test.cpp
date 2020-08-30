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

#include "kernels/Rsqrt.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<float> input_data, std::initializer_list<float> output_data)
{
  Tensor input_tensor{DataType::FLOAT32, input_shape, {}, ""};
  input_tensor.writeData(input_data.begin(), input_data.size() * sizeof(float));

  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  Rsqrt kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<float>(output_tensor),
              ::testing::ElementsAreArray(ArrayFloatNear(output_data)));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

TEST(RsqrtTest, SimpleRsqrt)
{
  Check(
      /*input_shape=*/{1, 2, 4, 1}, /*output_shape=*/{1, 2, 4, 1},
      /*input_data=*/
      {
          5, 4, 8, 2,     //
          6, 7.5, 9, 0.3, //
      },
      /*output_data=*/
      {
          0.44721360, 0.5, 0.35355339, 0.70710678,       //
          0.40824829, 0.36514837, 0.33333333, 1.8257419, //
      });
}

TEST(RsqrtTest, Input_Output_Type_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1}, {1.f});
  Tensor output_tensor = makeOutputTensor(DataType::S32);

  Rsqrt kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(RsqrtTest, Invalid_Input_Type_NEG)
{
  Tensor input_tensor = makeInputTensor<DataType::S64>({1}, {1});
  Tensor output_tensor = makeOutputTensor(DataType::S64);

  Rsqrt kernel(&input_tensor, &output_tensor);
  kernel.configure();
  EXPECT_ANY_THROW(kernel.execute());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
