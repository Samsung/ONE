/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/ArgMax.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T1, typename T2>
void Check(std::initializer_list<int32_t> input_shape,
           std::initializer_list<int32_t> dimension_shape,
           std::initializer_list<int32_t> output_shape, std::initializer_list<T1> input_data,
           std::initializer_list<int32_t> dimension_data, std::initializer_list<T2> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  constexpr DataType element_type = getElementType<T1>();
  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());
  Tensor dimension_tensor =
    makeInputTensor<DataType::S32>(dimension_shape, dimension_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(getElementType<T2>());

  ArgMaxParams params{};
  params.output_type = getElementType<T2>();
  ArgMax kernel(&input_tensor, &dimension_tensor, &output_tensor, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorData<T2>(output_tensor), ::testing::ElementsAreArray(output_data));
  EXPECT_THAT(extractTensorShape(output_tensor), output_shape);
}

template <typename T> class ArgMaxTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_SUITE(ArgMaxTest, DataTypes);

TYPED_TEST(ArgMaxTest, Simple)
{
  Check<TypeParam, int32_t>(/*input_shape=*/{1, 1, 1, 4}, /*dimension_shape=*/{},
                            /*output_shape=*/{1, 1, 1},
                            /*input_data=*/
                            {
                              1, 9, 7, 3, //
                            },
                            /*dimension_data=*/{3}, /*output_data=*/{1});
  Check<TypeParam, int64_t>(/*input_shape=*/{1, 1, 1, 4}, /*dimension_shape=*/{},
                            /*output_shape=*/{1, 1, 1},
                            /*input_data=*/
                            {
                              1, 9, 7, 3, //
                            },
                            /*dimension_data=*/{3}, /*output_data=*/{1});
}

TYPED_TEST(ArgMaxTest, MultiDimensions)
{
  Check<TypeParam, int32_t>(/*input_shape=*/{1, 1, 2, 4}, /*dimension_shape=*/{},
                            /*output_shape=*/{1, 1, 2},
                            /*input_data=*/
                            {
                              1, 2, 7, 8, //
                              1, 9, 7, 3, //
                            },
                            /*dimension_data=*/{3}, /*output_data=*/{3, 1});
  Check<TypeParam, int64_t>(/*input_shape=*/{1, 1, 2, 4}, /*dimension_shape=*/{},
                            /*output_shape=*/{1, 1, 2},
                            /*input_data=*/
                            {
                              1, 2, 7, 8, //
                              1, 9, 7, 3, //
                            },
                            /*dimension_data=*/{3}, /*output_data=*/{3, 1});
}

TEST(ArgMaxTest, UnsupportedType_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1, 1, 2, 4},
                                                           {
                                                             1, 2, 7, 8, //
                                                             1, 9, 7, 3, //
                                                           },
                                                           memory_manager.get());
  Tensor dimension_tensor = makeInputTensor<DataType::S32>({}, {3}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::U8);

  ArgMaxParams params{};
  params.output_type = DataType::U8;
  ArgMax kernel(&input_tensor, &dimension_tensor, &output_tensor, params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  EXPECT_ANY_THROW(kernel.execute());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
