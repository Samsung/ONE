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

#include "kernels/LeakyRelu.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T>
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<T> input_data, std::initializer_list<T> output_data, float alpha,
           DataType element_type)
{
  Tensor input_tensor{element_type, input_shape, {}, ""};
  input_tensor.writeData(input_data.begin(), input_data.size() * sizeof(T));

  Tensor output_tensor = makeOutputTensor(element_type);

  LeakyReluParams params{};
  params.alpha = alpha;

  LeakyRelu kernel(&input_tensor, &output_tensor, params);

  kernel.configure();
  kernel.execute();

  (void)output_shape;
  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
}

TEST(LeakReluTest, FloatSimple)
{
  Check<float>(/*input_shape=*/{2, 3}, /*output_shape=*/{2, 3}, /*input_data=*/
               {
                   0.0f, 1.0f, 3.0f,   // Row 1
                   1.0f, -1.0f, -2.0f, // Row 2
               },
               /*output_data=*/
               {
                   0.0f, 1.0f, 3.0f,   // Row 1
                   1.0f, -0.5f, -1.0f, // Row 2
               },
               /*alpha=*/0.5f, getElementType<float>());

  SUCCEED();
}

// TODO Uint8Simple
// Implement GetDequantizedOutput Function.
// Create Test for Uint8 Case

} // namespace
} // namespace kernels
} // namespace luci_interpreter
