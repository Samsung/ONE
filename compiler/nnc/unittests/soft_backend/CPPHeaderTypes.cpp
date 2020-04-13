/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <cstddef>
#include <initializer_list>
#include <functional>
#include <numeric>

#include "code_snippets/cpp_header_types.def"

#include "gtest/gtest.h"

template <class List> static inline void checkListShapeEq(List list, Shape shape, index_t volume)
{
  ASSERT_EQ(static_cast<size_t>(shape.getDims()), list.size());
  index_t idx = 0;
  for (auto el : list)
  {
    ASSERT_EQ(shape[idx], el);
    idx++;
  }
  ASSERT_EQ(shape.getNumElems(), volume);
}

TEST(SOFT_BACKEND, shape_and_index)
{
  auto list = {2, 3, 4};
  index_t volume = std::accumulate(list.begin(), list.end(), 1, std::multiplies<index_t>());
  Shape s1(list);
  checkListShapeEq(list, s1, volume);
// This check must be performed only if assertions are enabled
#ifndef NDEBUG
  ASSERT_DEATH(s1[list.size()], "");
#endif

  Shape s2(s1);
  checkListShapeEq(list, s2, volume);

  Shape s3{1};
  ASSERT_EQ(s3.getNumElems(), 1);
  ASSERT_EQ(s3.getDims(), 1);
  s3 = s1;
  checkListShapeEq(list, s3, volume);

  s3.setDims(4);
  s3[3] = 2;
  ASSERT_EQ(s3.getNumElems(), volume * 2);
  s3.setDims(3);
  ASSERT_EQ(s3.getNumElems(), volume);
}

TEST(SOFT_BACKEND, tensor)
{
  // test reshape
  Tensor t1;
  ASSERT_EQ(t1.getShape().getNumElems(), 1);
  const index_t tensor1_height = 2;
  const index_t tensor1_width = 4;
  t1.reshape(Shape{tensor1_height, tensor1_width});
  ASSERT_EQ(t1.getShape().getNumElems(), tensor1_height * tensor1_width);
  // test at functions
  float expected_sum = 0;
  for (index_t i = 0; i < tensor1_height; ++i)
    for (index_t j = 0; j < tensor1_width; ++j)
    {
      index_t elem = (i + 1) * (j + 1);
      expected_sum += elem;
      t1.at({i, j}) = elem;
    }
  float sum = 0;
  for (index_t i = 0; i < tensor1_height; ++i)
    for (index_t j = 0; j < tensor1_width; ++j)
    {
      sum += t1.at({i, j});
    }
  ASSERT_EQ(sum, expected_sum);

  // test construction with shape
  const index_t tensor2_height = 3;
  const index_t tensor2_width = 4;
  Tensor t2({tensor2_height, tensor2_width});
  ASSERT_EQ(t2.getShape().getNumElems(), tensor2_height * tensor2_width);

  // test unmanaged tensor
  const index_t tensor3_depth = 2;
  const index_t tensor3_height = 2;
  const index_t tensor3_width = 3;
  std::vector<float> data({1.0, 2.0, 4.0});
  data.resize(tensor3_depth * tensor3_height * tensor3_width);
  float *data_ptr = data.data();
  Tensor t3(Shape({tensor3_depth, tensor3_height, tensor3_width}), data_ptr);
  ASSERT_EQ(t3.getShape().getNumElems(), tensor3_depth * tensor3_height * tensor3_width);
  sum = 0;
  for (index_t k = 0; k < tensor3_depth; ++k)
    for (index_t i = 0; i < tensor3_height; ++i)
      for (index_t j = 0; j < tensor3_width; ++j)
      {
        sum += t3.at({k, i, j});
      }
  ASSERT_EQ(sum, std::accumulate(data_ptr, data_ptr + t3.getShape().getNumElems(), 0.0f));

  // test tensor copy
  const index_t t4Width = 4;
  Tensor t4({t4Width});
  t4 = t3;
  for (index_t k = 0; k < tensor3_depth; ++k)
    for (index_t i = 0; i < tensor3_height; ++i)
      for (index_t j = 0; j < tensor3_height; ++j)
      {
        ASSERT_EQ(t3.at({k, i, j}), t4.at({k, i, j}));
      }
}
