/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "kernels/SplitV.h"
#include "kernels/TestUtils.h"

namespace luci_interpreter
{
namespace kernels
{
namespace
{

using namespace testing;

template <typename T>
void Check(int axis, std::initializer_list<int32_t> splits_size,
           std::initializer_list<int32_t> input_shape, std::initializer_list<T> input_data,
           std::vector<std::vector<T>> output_data)
{
  constexpr DataType element_type = getElementType<T>();

  auto num_splits = static_cast<int32_t>(splits_size.size());
  Tensor input_tensor = makeInputTensor<element_type>(input_shape, input_data);
  Tensor sizes_tensor = makeInputTensor<DataType::S32>({num_splits}, splits_size);
  Tensor axis_tensor = makeInputTensor<DataType::S32>({}, {axis});

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

  SplitV kernel(&input_tensor, &sizes_tensor, &axis_tensor, std::move(output_tensor_ptrs));
  kernel.configure();
  kernel.execute();

  for (int i = 0; i < num_splits; ++i)
  {
    auto tmp = extractTensorData<T>(output_tensors[i]);
    EXPECT_THAT(extractTensorData<T>(output_tensors[i]),
                ::testing::ElementsAreArray(output_data[i]));
  }
}

template <typename T> class SplitVTest : public ::testing::Test
{
};

using DataTypes = ::testing::Types<float, uint8_t, int16_t>;
TYPED_TEST_CASE(SplitVTest, DataTypes);

TYPED_TEST(SplitVTest, ThreeDimensional)
{
  Check<TypeParam>(
    /*axis=*/0, /*splits_size=*/{1, 2}, {3, 3, 3},
    {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
    {
      {1, 2, 3, 4, 5, 6, 7, 8, 9},                                             //
      {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27} //
    });
  Check<TypeParam>(
    /*axis=*/1, /*splits_size=*/{1, 2}, {3, 3, 3},
    {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
    {
      {1, 2, 3, 10, 11, 12, 19, 20, 21},                                 //
      {4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27} //
    });
  Check<TypeParam>(
    /*axis=*/2, /*splits_size=*/{1, 2}, {3, 3, 3},
    {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
     15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27},
    {
      {1, 4, 7, 10, 13, 16, 19, 22, 25},                                 //
      {2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27} //
    });
}

} // namespace
} // namespace kernels
} // namespace luci_interpreter
