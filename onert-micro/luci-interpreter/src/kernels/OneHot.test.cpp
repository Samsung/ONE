/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/OneHot.h"
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
void Check(std::initializer_list<int32_t> input_shape, std::initializer_list<int32_t> output_shape,
           std::initializer_list<T1> input_data, std::initializer_list<int32_t> depth_data,
           std::initializer_list<T2> on_value_data, std::initializer_list<T2> off_value_data,
           int32_t axis, std::initializer_list<T2> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  constexpr auto input_type = getElementType<T1>();
  constexpr auto output_type = getElementType<T2>();

  Tensor input_tensor = makeInputTensor<input_type>(input_shape, input_data, memory_manager.get());
  Tensor depth_tensor = makeInputTensor<DataType::S32>({}, depth_data, memory_manager.get());
  Tensor on_value_tensor = makeInputTensor<output_type>({}, on_value_data, memory_manager.get());
  Tensor off_value_tensor = makeInputTensor<output_type>({}, off_value_data, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(output_type);

  OneHotParams params{};
  params.axis = axis;

  OneHot kernel(&input_tensor, &depth_tensor, &on_value_tensor, &off_value_tensor, &output_tensor,
                params);
  kernel.configure();
  memory_manager->allocate_memory(output_tensor);
  kernel.execute();

  EXPECT_THAT(extractTensorShape(output_tensor), output_shape);
  EXPECT_THAT(extractTensorData<T2>(output_tensor), ::testing::ElementsAreArray(output_data));
}

template <typename T> class OneHotTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t, int16_t>;
TYPED_TEST_SUITE(OneHotTest, DataTypes);

TYPED_TEST(OneHotTest, BasicPattern)
{
  // axis 0
  Check<int32_t, TypeParam>(/*input_shape=*/{2, 3}, /*output_shape=*/{4, 2, 3},
                            /*input_data=*/
                            {
                              0, 3, 5, //
                              7, 3, 0, //
                            },
                            /*depth_data=*/{4}, /*on_value_data=*/{1}, /*off_value_data=*/{0},
                            /*axis=*/0,
                            /*output_data=*/
                            {
                              1, 0, 0, //
                              0, 0, 1, //

                              0, 0, 0, //
                              0, 0, 0, //

                              0, 0, 0, //
                              0, 0, 0, //

                              0, 1, 0, //
                              0, 1, 0, //
                            });
  // axis 1
  Check<int32_t, TypeParam>(/*input_shape=*/{2, 3}, /*output_shape=*/{2, 4, 3},
                            /*input_data=*/
                            {
                              0, 3, 5, //
                              7, 3, 0, //
                            },
                            /*depth_data=*/{4}, /*on_value_data=*/{1}, /*off_value_data=*/{0},
                            /*axis=*/1,
                            /*output_data=*/
                            {
                              1, 0, 0, //
                              0, 0, 0, //
                              0, 0, 0, //
                              0, 1, 0, //

                              0, 0, 1, //
                              0, 0, 0, //
                              0, 0, 0, //
                              0, 1, 0, //
                            });
  // axis -1
  Check<int32_t, TypeParam>(/*input_shape=*/{2, 3}, /*output_shape=*/{2, 3, 4},
                            /*input_data=*/
                            {
                              0, 3, 5, //
                              7, 3, 0, //
                            },
                            /*depth_data=*/{4}, /*on_value_data=*/{1}, /*off_value_data=*/{0},
                            /*axis=*/-1,
                            /*output_data=*/
                            {
                              1, 0, 0, 0, //
                              0, 0, 0, 1, //
                              0, 0, 0, 0, //

                              0, 0, 0, 0, //
                              0, 0, 0, 1, //
                              1, 0, 0, 0, //
                            });
}

TEST(OneHotTest, UnsupportedInputType_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  // input type should be integer
  Tensor input_tensor = makeInputTensor<DataType::FLOAT32>({1}, {0}, memory_manager.get());

  Tensor depth_tensor = makeInputTensor<DataType::S32>({}, {1}, memory_manager.get());
  Tensor on_value_tensor = makeInputTensor<DataType::FLOAT32>({}, {1.0}, memory_manager.get());
  Tensor off_value_tensor = makeInputTensor<DataType::FLOAT32>({}, {0.0}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  OneHotParams params = {-1};

  OneHot kernel(&input_tensor, &depth_tensor, &on_value_tensor, &off_value_tensor, &output_tensor,
                params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(OneHotTest, OutputTypeMismatch_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor = makeInputTensor<DataType::S32>({1}, {0}, memory_manager.get());
  Tensor depth_tensor = makeInputTensor<DataType::S32>({}, {1}, memory_manager.get());

  // type of on_value, off_value and output_tensor should be same
  Tensor on_value_tensor = makeInputTensor<DataType::FLOAT32>({}, {1.0}, memory_manager.get());
  Tensor off_value_tensor = makeInputTensor<DataType::FLOAT32>({}, {0.0}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::S16);

  OneHotParams params = {-1};

  OneHot kernel(&input_tensor, &depth_tensor, &on_value_tensor, &off_value_tensor, &output_tensor,
                params);
  EXPECT_ANY_THROW(kernel.configure());
}

TEST(OneHotTest, InvalidAxis_NEG)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  Tensor input_tensor = makeInputTensor<DataType::S32>({1}, {0}, memory_manager.get());
  Tensor depth_tensor = makeInputTensor<DataType::S32>({}, {1}, memory_manager.get());
  Tensor on_value_tensor = makeInputTensor<DataType::FLOAT32>({}, {1.0}, memory_manager.get());
  Tensor off_value_tensor = makeInputTensor<DataType::FLOAT32>({}, {0.0}, memory_manager.get());
  Tensor output_tensor = makeOutputTensor(DataType::FLOAT32);

  // axis should be in [-1, input_shape.rank]
  OneHotParams params = {-2};

  OneHot kernel(&input_tensor, &depth_tensor, &on_value_tensor, &off_value_tensor, &output_tensor,
                params);
  EXPECT_ANY_THROW(kernel.configure());
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
