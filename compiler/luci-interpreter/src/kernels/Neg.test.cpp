/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Neg.h"
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
           std::initializer_list<T> input_data, std::initializer_list<T> output_data)
{
  constexpr DataType element_type = getElementType<T>();
  Tensor input_tensor = makeInputTensor<element_type>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(element_type);

  Neg kernel(&input_tensor, &output_tensor);

  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

TEST(NegTest, FloatSimple)
{
  Check<float>(/*input_shape=*/{2, 3},
               /*output_shape=*/{2, 3},
               /*input_data=*/
               {
                 0.0f, 1.0f, 3.0f,   // Row 1
                 1.0f, -1.0f, -2.0f, // Row 2
               },
               /*output_data=*/
               {
                 0.0f, -1.0f, -3.0f, // Row 1
                 -1.0f, 1.0f, 2.0f,  // Row 2
               });

  SUCCEED();
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
