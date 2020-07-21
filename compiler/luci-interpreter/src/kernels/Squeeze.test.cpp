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

#include "kernels/Squeeze.h"
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
           std::initializer_list<T> input_data, std::initializer_list<T> output_data,
           DataType element_type, std::vector<int32_t> squeeze_dims)
{
  Tensor input_tensor{element_type, input_shape, {}, ""};
  input_tensor.writeData(input_data.begin(), input_data.size() * sizeof(T));
  Tensor output_tensor = makeOutputTensor(element_type);

  SqueezeParams params{};
  for (size_t i = 0; i < squeeze_dims.size(); i++)
  {
    params.squeeze_dims.push_back(squeeze_dims.at(i));
  }

  Squeeze kernel(&input_tensor, &output_tensor, params);
  kernel.configure();
  kernel.execute();

  EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), ::testing::ElementsAreArray(output_shape));
}

template <typename T> class SqueezeTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_CASE(SqueezeTest, DataTypes);

TYPED_TEST(SqueezeTest, TotalTest)
{
  Check<TypeParam>(
      /*input_shape=*/{1, 24, 1}, /*output_shape=*/{24},
      /*input_data=*/{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
      /*output_data=*/{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
      getElementType<TypeParam>(), {-1, 0});
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
