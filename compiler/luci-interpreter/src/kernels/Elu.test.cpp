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

#include "kernels/Elu.h"
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
           DataType element_type)
{
  Tensor input_tensor{element_type, input_shape, {}, ""};
  input_tensor.writeData(input_data.begin(), input_data.size() * sizeof(T));

  Tensor output_tensor = makeOutputTensor(element_type);

  Elu kernel(&input_tensor, &output_tensor);
  kernel.configure();
  kernel.execute();

  if (element_type == DataType::FLOAT32)
    EXPECT_THAT(extractTensorData<T>(output_tensor),
                ::testing::ElementsAreArray(ArrayFloatNear(output_data)));
  else
    EXPECT_THAT(extractTensorData<T>(output_tensor), ::testing::ElementsAreArray(output_data));
}

template <typename T> class EluTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float>;
TYPED_TEST_CASE(EluTest, DataTypes);

TYPED_TEST(EluTest, SimpleElu)
{
  Check<TypeParam>(
      /*input_shape=*/{1, 2, 4, 1}, /*output_shape=*/{1, 2, 4, 1},
      /*input_data=*/
      {
          0, -6, 2, -4,    //
          3, -2, 10, -0.1, //
      },
      /*output_data=*/
      {
          0.0, -0.997521, 2.0, -0.981684,   //
          3.0, -0.864665, 10.0, -0.0951626, //
      },
      getElementType<TypeParam>());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
