/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
#if 0
#include "kernels/Split.h"
#include "kernels/TestUtils.h"
#include "luci_interpreter/TestMemoryManager.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T>
void Check(int axis, int num_splits, std::initializer_list<int32_t> input_shape,
           std::initializer_list<int32_t> output_shape, std::initializer_list<T> input_data,
           std::vector<std::vector<T>> output_data)
{
  std::unique_ptr<IMemoryManager> memory_manager = std::make_unique<TestMemoryManager>();

  constexpr DataType element_type = getElementType<T>();
  Tensor axis_tensor = makeInputTensor<DataType::S32>({}, {axis}, memory_manager.get());
  Tensor input_tensor =
    makeInputTensor<element_type>(input_shape, input_data, memory_manager.get());

  std::vector<Tensor> output_tensors;
  output_tensors.reserve(num_splits);
  for (int i = 0; i < num_splits; ++i)
  {
    output_tensors.emplace_back(makeOutputTensor(element_type));
  }

  std::vector<Tensor *> output_tensor_ptrs(num_splits);
  for (int i = 0; i < num_splits; ++i)
  {
    output_tensor_ptrs[i] = &output_tensors[i];
  }

  Split kernel(&axis_tensor, &input_tensor, std::move(output_tensor_ptrs));
  kernel.configure();
  for (int i = 0; i < num_splits; ++i)
  {
    memory_manager->allocate_memory(output_tensors[i]);
  }
  kernel.execute();

  for (int i = 0; i < num_splits; ++i)
  {
    EXPECT_THAT(extractTensorData<T>(output_tensors[i]),
                ::testing::ElementsAreArray(output_data[i]));
  }
}

template <typename T> class SplitTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t>;
TYPED_TEST_SUITE(SplitTest, DataTypes);

TYPED_TEST(SplitTest, FourDimensional)
{
  Check<TypeParam>(/*axis=*/0, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                   {
                     {1, 2, 3, 4, 5, 6, 7, 8},        //
                     {9, 10, 11, 12, 13, 14, 15, 16}, //
                   });
  Check<TypeParam>(
    /*axis=*/1, /*num_splits=*/2, {2, 2, 2, 2}, {2, 1, 2, 2},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {
      {1, 2, 3, 4, 9, 10, 11, 12},  //
      {5, 6, 7, 8, 13, 14, 15, 16}, //
    });
  Check<TypeParam>(
    /*axis=*/2, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 1, 2},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {
      {1, 2, 5, 6, 9, 10, 13, 14},  //
      {3, 4, 7, 8, 11, 12, 15, 16}, //
    });
  Check<TypeParam>(
    /*axis=*/3, /*num_splits=*/2, {2, 2, 2, 2}, {2, 2, 2, 1},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {
      {1, 3, 5, 7, 9, 11, 13, 15},  //
      {2, 4, 6, 8, 10, 12, 14, 16}, //
    });
}

TYPED_TEST(SplitTest, OneDimensional)
{
  Check<TypeParam>(
    /*axis=*/0, /*num_splits=*/8, {8}, {1}, {1, 2, 3, 4, 5, 6, 7, 8},
    {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}});
}

TYPED_TEST(SplitTest, NegativeAxis)
{
  Check<TypeParam>(
    /*axis=*/-4, /*num_splits=*/2, {2, 2, 2, 2}, {1, 2, 2, 2},
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
    {
      {1, 2, 3, 4, 5, 6, 7, 8}, //
      {9, 10, 11, 12, 13, 14, 15, 16},
    });
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
#endif
