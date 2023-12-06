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

#include <luci_compute/FullyConnected.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

class FullyConnectedTest : public ::testing::Test
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
  luci::compute::FullyConnected _fc;
};

TEST_F(FullyConnectedTest, prepare_compute)
{
  auto input_shape = tensor_shape({3, 2, 2, 1});
  std::vector<float> input_data{
    -3, -5, 5,  4,  //
    9,  -2, -3, -2, //
    -4, 9,  -8, 1,  //
  };
  auto weights_shape = tensor_shape({3, 6});
  std::vector<float> weights_data{
    -3, -7, 4, -4, -6, 4,  //
    3,  5,  2, 3,  -3, -8, //
    -3, 7,  4, 9,  0,  -5, //
  };
  auto bias_shape = tensor_shape({3});
  std::vector<float> bias_data{-1, -5, -8};

  auto &params = _fc.params();
  params.weights_format = luci::compute::FullyConnectedWeightsFormat::kDefault;

  _fc.input(input_shape, input_data.data());
  _fc.weights(weights_shape, weights_data.data());
  _fc.bias(bias_shape, bias_data.data());
  _fc.fused_act_func(luci::compute::FusedActFunc::RELU);

  EXPECT_TRUE(_fc.prepare());

  auto output_shape = _fc.output_shape();
  auto output_count = loco::element_count(&output_shape);
  std::vector<float> output_data_vector;
  output_data_vector.resize(output_count);

  _fc.output(output_data_vector.data());

  ASSERT_NO_THROW(_fc.compute());

  std::vector<float> ref_output_data{
    0,  0,  32, //
    22, 11, 47, //
  };
  std::vector<uint32_t> ref_output_shape{2, 3};
  std::vector<uint32_t> output_shape_vector = vector_shape(output_shape);

  EXPECT_THAT(output_data_vector, ref_output_data);
  EXPECT_THAT(output_shape_vector, ref_output_shape);
}

TEST_F(FullyConnectedTest, prepare_invalid_rank_NEG)
{
  auto input_shape = tensor_shape({3});
  std::vector<float> input_data{-3, -5, 5};
  auto weights_shape = tensor_shape({3}); // expect rank 2
  std::vector<float> weights_data{-3, -7, 4};
  auto bias_shape = tensor_shape({3});
  std::vector<float> bias_data{-1, -5, -8};

  _fc.input(input_shape, input_data.data());
  _fc.weights(weights_shape, weights_data.data());
  _fc.bias(bias_shape, bias_data.data());
  _fc.fused_act_func(luci::compute::FusedActFunc::RELU);

  EXPECT_FALSE(_fc.prepare());
}

TEST_F(FullyConnectedTest, prepare_invalid_shape_NEG)
{
  auto input_shape = tensor_shape({3, 2, 2, 1});
  std::vector<float> input_data{
    -3, -5, 5,  4,  //
    9,  -2, -3, -2, //
    -4, 9,  -8, 1,  //
  };
  auto weights_shape = tensor_shape({3, 5}); // expect 3, 6
  std::vector<float> weights_data{
    -3, -7, 4, -4, -6, //
    3,  5,  2, 3,  -3, //
    -3, 7,  4, 9,  0,  //
  };
  auto bias_shape = tensor_shape({3});
  std::vector<float> bias_data{-1, -5, -8};

  _fc.input(input_shape, input_data.data());
  _fc.weights(weights_shape, weights_data.data());
  _fc.bias(bias_shape, bias_data.data());
  _fc.fused_act_func(luci::compute::FusedActFunc::RELU);

  EXPECT_FALSE(_fc.prepare());
}
