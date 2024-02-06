/* Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConvertValues.h"

#include <luci_compute/StridedSlice.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

class StridedSliceTest : public ::testing::Test
{
protected:
  loco::TensorShape tensor_shape(const std::initializer_list<uint32_t> shape)
  {
    loco::TensorShape tensor_shape;
    tensor_shape.rank(shape.size());
    uint32_t i = 0;
    for (auto it = shape.begin(); it != shape.end(); ++it, ++i)
      tensor_shape.dim(i) = *it;
    return tensor_shape;
  }

  std::vector<uint32_t> vector_shape(const loco::TensorShape &tensor_shape)
  {
    std::vector<uint32_t> shape;
    for (uint32_t r = 0; r < tensor_shape.rank(); ++r)
      shape.push_back(tensor_shape.dim(r).value());
    return shape;
  }

protected:
  luci::compute::StridedSlice<int32_t> _strided_slice;
};

TEST_F(StridedSliceTest, prepare_compute)
{
  auto input_shape = tensor_shape({1, 4, 4, 1});
  std::vector<int32_t> input_data{
    1,  2,  3,  4,  //
    5,  6,  7,  8,  //
    9,  10, 11, 12, //
    13, 14, 15, 16, //
  };
  auto begin_shape = tensor_shape({4});
  std::vector<int32_t> begin_data{
    0, 0, 0, 0, //
  };
  auto end_shape = tensor_shape({4});
  std::vector<int32_t> end_data{
    1, 4, 4, 1, //
  };
  auto strides_shape = tensor_shape({4});
  std::vector<int32_t> strides_data{
    1, 2, 2, 1, //
  };

  auto &params = _strided_slice.params();
  params.start_indices_count = 4;
  params.start_indices[0] = 0;
  params.start_indices[1] = 0;
  params.start_indices[2] = 0;
  params.start_indices[3] = 0;
  params.stop_indices_count = 4;
  params.stop_indices[0] = 1;
  params.stop_indices[1] = 4;
  params.stop_indices[2] = 4;
  params.stop_indices[3] = 1;
  params.strides_count = 4;
  params.strides[0] = 1;
  params.strides[1] = 2;
  params.strides[2] = 2;
  params.strides[3] = 1;

  params.begin_mask = 0;
  params.end_mask = 0;
  params.ellipsis_mask = 0;
  params.new_axis_mask = 0;
  params.shrink_axis_mask = 0;

  _strided_slice.input(input_shape, input_data.data());
  _strided_slice.begin(begin_shape, begin_data.data());
  _strided_slice.end(end_shape, end_data.data());
  _strided_slice.strides(strides_shape, strides_data.data());

  EXPECT_TRUE(_strided_slice.prepare());

  auto output_shape = _strided_slice.output_shape();
  auto output_count = loco::element_count(&output_shape);
  std::vector<int32_t> output_data_vector;
  output_data_vector.resize(output_count);

  _strided_slice.output(output_data_vector.data());

  ASSERT_NO_THROW(_strided_slice.compute());

  std::vector<int32_t> ref_output_data{
    1, 3, 9, 11, //
  };
  std::vector<uint32_t> ref_output_shape{1, 2, 2, 1};
  std::vector<uint32_t> output_shape_vector = vector_shape(output_shape);

  EXPECT_THAT(output_data_vector, ref_output_data);
  EXPECT_THAT(output_shape_vector, ref_output_shape);
}

TEST_F(StridedSliceTest, prepare_compute_2)
{
  auto input_shape = tensor_shape({4});
  std::vector<int32_t> input_data{
    10, 20, 30, 40, //
  };
  auto begin_shape = tensor_shape({1});
  std::vector<int32_t> begin_data{
    0, //
  };
  auto end_shape = tensor_shape({1});
  std::vector<int32_t> end_data{
    4, //
  };
  auto strides_shape = tensor_shape({1});
  std::vector<int32_t> strides_data{
    2, //
  };

  auto &params = _strided_slice.params();
  params.start_indices_count = 1;
  params.start_indices[0] = 0;
  params.stop_indices_count = 1;
  params.stop_indices[0] = 4;
  params.strides_count = 1;
  params.strides[0] = 2;

  params.begin_mask = 0;
  params.end_mask = 0;
  params.ellipsis_mask = 0;
  params.new_axis_mask = 0;
  params.shrink_axis_mask = 0;

  _strided_slice.input(input_shape, input_data.data());
  _strided_slice.begin(begin_shape, begin_data.data());
  _strided_slice.end(end_shape, end_data.data());
  _strided_slice.strides(strides_shape, strides_data.data());

  EXPECT_TRUE(_strided_slice.prepare());

  auto output_shape = _strided_slice.output_shape();
  auto output_count = loco::element_count(&output_shape);
  std::vector<int32_t> output_data_vector;
  output_data_vector.resize(output_count);

  _strided_slice.output(output_data_vector.data());

  ASSERT_NO_THROW(_strided_slice.compute());

  std::vector<int32_t> ref_output_data{
    10, 30, //
  };
  std::vector<uint32_t> ref_output_shape{2};
  std::vector<uint32_t> output_shape_vector = vector_shape(output_shape);

  EXPECT_THAT(output_data_vector, ref_output_data);
  EXPECT_THAT(output_shape_vector, ref_output_shape);
}
