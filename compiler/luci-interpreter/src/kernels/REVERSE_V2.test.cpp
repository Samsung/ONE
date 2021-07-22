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

#include "kernels/REVERSE_V2.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T> class ReverseTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(ReverseTest, DataTypes);

TYPED_TEST(ReverseTest, MultiDimensions)
{
  // TypeParam
  std::vector<TypeParam> input_data{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  Shape input_shape{4, 3, 2};
  std::vector<int32_t> axis_data{1};
  Shape axis_shape{1};

  std::vector<TypeParam> output_data{5,  6,  3,  4,  1,  2,  11, 12, 9,  10, 7,  8,
                                     17, 18, 15, 16, 13, 14, 23, 24, 21, 22, 19, 20};
  std::vector<int32_t> output_shape{4, 3, 2};

  Tensor input_tensor = makeInputTensor<getElementType<TypeParam>()>(input_shape, input_data);
  Tensor axis_tensor = makeInputTensor<DataType::S32>(axis_shape, axis_data);

  Tensor output_tensor = makeOutputTensor(getElementType<TypeParam>());

  Reverse kernel = Reverse(&input_tensor, &axis_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<TypeParam>(output_tensor),
              ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
