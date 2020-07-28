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

#include "kernels/DepthToSpace.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T> class DepthToSpaceTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(DepthToSpaceTest, DataTypes);

TYPED_TEST(DepthToSpaceTest, SimpleCase)
{
  std::vector<TypeParam> input_data{1, 2, 3, 4, 5, 6, 7, 8};
  Shape input_shape{1, 1, 2, 4};
  std::vector<TypeParam> output_data{1, 2, 5, 6, 3, 4, 7, 8};
  std::vector<int32_t> output_shape{1, 2, 4, 1};

  Tensor input_tensor = makeInputTensor<getElementType<TypeParam>()>(input_shape, input_data);
  Tensor output_tensor = makeOutputTensor(getElementType<TypeParam>());

  DepthToSpaceParams params{};
  params.block_size = 2;

  DepthToSpace kernel = DepthToSpace(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<TypeParam>(output_tensor),
              ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
