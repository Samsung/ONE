/* Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <luci_compute/DepthwiseConv2D.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

class DepthwiseConv2DTest : public ::testing::Test
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
  luci::compute::DepthwiseConv2D _dwconv2d;
};

TEST_F(DepthwiseConv2DTest, prepare_compute)
{
  auto input_shape = tensor_shape({1, 4, 2, 2});
  std::vector<float> input_data{
    1,  2,  7,  8,  //
    3,  4,  9,  10, //
    5,  6,  11, 12, //
    13, 14, 15, 16, //
  };
  auto filter_shape = tensor_shape({1, 2, 2, 4});
  std::vector<float> filter_data{
    1,  2,   3,   4,   //
    -9, 10,  -11, 12,  //
    5,  6,   7,   8,   //
    13, -14, 15,  -16, //
  };
  auto bias_shape = tensor_shape({4});
  std::vector<float> bias_data{1, 2, 3, 4};

  auto &params = _dwconv2d.params();
  params.padding_type = luci::compute::PaddingType::kValid;
  params.stride_height = 2;
  params.stride_width = 1;
  params.dilation_height_factor = 1;
  params.dilation_width_factor = 1;
  params.depth_multiplier = 2;

  _dwconv2d.input(input_shape, input_data.data());
  _dwconv2d.filter(filter_shape, filter_data.data());
  _dwconv2d.bias(bias_shape, bias_data.data());
  _dwconv2d.fused_act_func(luci::compute::FusedActFunc::RELU);

  EXPECT_TRUE(_dwconv2d.prepare());

  auto output_shape = _dwconv2d.output_shape();
  auto output_count = loco::element_count(&output_shape);
  std::vector<float> output_data_vector;
  output_data_vector.resize(output_count);

  _dwconv2d.output(output_data_vector.data());

  ASSERT_NO_THROW(_dwconv2d.compute());

  std::vector<float> ref_output_data{
    71,  0, 99,  0,  //
    167, 0, 227, 28, //
  };
  std::vector<uint32_t> ref_output_shape{1, 2, 1, 4};
  std::vector<uint32_t> output_shape_vector = vector_shape(output_shape);

  EXPECT_THAT(output_data_vector, ref_output_data);
  EXPECT_THAT(output_shape_vector, ref_output_shape);
}

TEST_F(DepthwiseConv2DTest, prepare_invalid_rank_NEG)
{
  auto input_shape = tensor_shape({2}); // expect rank 4
  std::vector<float> input_data{1, 2};
  auto filter_shape = tensor_shape({2});
  std::vector<float> filter_data{1, 2};
  auto bias_shape = tensor_shape({1});
  std::vector<float> bias_data{1};

  _dwconv2d.input(input_shape, input_data.data());
  _dwconv2d.filter(filter_shape, filter_data.data());
  _dwconv2d.bias(bias_shape, bias_data.data());
  _dwconv2d.fused_act_func(luci::compute::FusedActFunc::RELU);

  EXPECT_FALSE(_dwconv2d.prepare());
}

TEST_F(DepthwiseConv2DTest, prepare_invalid_shape_NEG)
{
  auto input_shape = tensor_shape({1, 4, 2, 2});
  std::vector<float> input_data{
    1,  2,  7,  8,  //
    3,  4,  9,  10, //
    5,  6,  11, 12, //
    13, 14, 15, 16, //
  };
  auto filter_shape = tensor_shape({1, 2, 2, 3}); // expect ,,, 4
  std::vector<float> filter_data{
    1,  2,  3,   4,  //
    -9, 10, -11, 12, //
    5,  6,  7,   8,  //
  };
  auto bias_shape = tensor_shape({4});
  std::vector<float> bias_data{1, 2, 3, 4};

  _dwconv2d.input(input_shape, input_data.data());
  _dwconv2d.filter(filter_shape, filter_data.data());
  _dwconv2d.bias(bias_shape, bias_data.data());
  _dwconv2d.fused_act_func(luci::compute::FusedActFunc::RELU);

  EXPECT_FALSE(_dwconv2d.prepare());
}
