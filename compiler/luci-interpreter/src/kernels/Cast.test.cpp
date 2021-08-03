/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/Cast.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/SimpleMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T1, typename T2>
void Check(std::initializer_list<int32_t> shape, std::initializer_list<T1> input_data,
           std::initializer_list<T2> output_data)
{
  std::unique_ptr<MManager> memory_manager = std::make_unique<SimpleMManager>();
  constexpr DataType input_type = getElementType<T1>();
  constexpr DataType output_type = getElementType<T2>();

  Tensor input_tensor = makeInputTensor<input_type>(shape, input_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(output_type);

  Cast kernel(&input_tensor, &output_tensor);
  kernel.configure();
  memory_manager->allocate_memory(&output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<T2>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), shape);
}

template <typename T> class CastTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<uint8_t, int32_t, int64_t>;
TYPED_TEST_CASE(CastTest, DataTypes);

TYPED_TEST(CastTest, FloatToInt)
{
  Check<float, TypeParam>(/*shape=*/{1, 1, 1, 4},
                          /*input_data=*/
                          {
                            1.43f, 9.99f, 7.0f, 3.12f, //
                          },
                          /*output_data=*/
                          {
                            1, 9, 7, 3, //
                          });
  Check<TypeParam, TypeParam>(/*shape=*/{1, 1, 1, 4},
                              /*input_data=*/
                              {
                                1, 9, 7, 3, //
                              },
                              /*output_data=*/
                              {
                                1, 9, 7, 3, //
                              });
}

TEST(CastTest, UnsupportedType_NEG)
{
  std::unique_ptr<MManager> memory_manager = std::make_unique<SimpleMManager>();
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 1, 2, 4},
                                                           {
                                                             1, 2, 7, 8, //
                                                             1, 9, 7, 3, //
                                                           },
                                                           memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::Unknown);

  Cast kernel(&input_tensor, &output_tensor);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
