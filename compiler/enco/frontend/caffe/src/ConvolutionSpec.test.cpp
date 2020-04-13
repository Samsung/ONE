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

#include "ConvolutionSpec.h"
#include "Importer.h"

#include <nncc/core/ADT/tensor/Shape.h>

#include <caffe/net.hpp>

#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

using namespace nncc::core::ADT;

#define STRING(content) #content

namespace
{
class ConvolutionSpecTest : public ::testing::Test
{
protected:
  tensor::Shape as_tensor_shape(const std::vector<int> &dims)
  {
    const uint32_t rank = dims.size();

    tensor::Shape res;

    res.resize(rank);

    for (uint32_t axis = 0; axis < rank; ++axis)
    {
      res.dim(axis) = dims.at(axis);
    }

    return res;
  }

  bool load(const std::string &prototxt, ::caffe::NetParameter &param)
  {
    std::stringstream ss{prototxt};

    return from_txt(ss, param);
  }
};
} // namespace

TEST_F(ConvolutionSpecTest, ifm_shape)
{
  ::caffe::ConvolutionParameter param;
  ConvolutionSpec spec{param};

  const tensor::Shape ifm_shape{1, 3, 244, 244};

  spec.ifm_shape(ifm_shape);

  ASSERT_EQ(spec.ifm_shape(), ifm_shape);
  ASSERT_EQ(spec.num_batch_axes(), 1);
  ASSERT_EQ(spec.num_spatial_axes(), 2);
}

namespace
{
// clang-format off
const char *conv_0 = STRING(
layer {
  name: "data"
  type : "Input"
  top : "data"
  input_param { shape: { dim: 1 dim : 3 dim : 244 dim : 244 } }
}
layer{
  name : "conv"
  type : "Convolution"
  bottom : "data"
  top : "conv"
  convolution_param {
    bias_term : false
    num_output : 1
    kernel_size : 1
  }
});
// clang-format on
} // namespace

TEST_F(ConvolutionSpecTest, conv_0)
{
  ::caffe::NetParameter param;

  ASSERT_TRUE(load(conv_0, param));

  ::caffe::Net<float> net{param};

  const tensor::Shape ifm_shape{1, 3, 244, 244};
  ConvolutionSpec spec{param.layer(1).convolution_param()};

  spec.ifm_shape(ifm_shape);

  // Check 'ker_shape'
  {
    auto expected = as_tensor_shape(net.layer_by_name("conv")->blobs().at(0)->shape());
    auto obtained = spec.ker_shape();

    ASSERT_EQ(expected, obtained);
  }

  // Check 'ofm_shape'
  {
    auto expected = as_tensor_shape(net.blob_by_name("conv")->shape());
    auto obtained = spec.ofm_shape();

    ASSERT_EQ(expected, obtained);
  }
}

namespace
{
// clang-format off
const char *conv_1 = STRING(
layer {
  name: "data"
  type : "Input"
  top : "data"
  input_param { shape: { dim: 1 dim : 3 dim : 244 dim : 244 } }
}
layer{
  name : "conv"
  type : "Convolution"
  bottom : "data"
  top : "conv"
  convolution_param {
    bias_term : false
    num_output : 1
    kernel_size : 1
    kernel_size : 3
  }
});
// clang-format on
} // namespace

TEST_F(ConvolutionSpecTest, conv_1)
{
  ::caffe::NetParameter param;

  ASSERT_TRUE(load(conv_1, param));

  ::caffe::Net<float> net{param};

  const tensor::Shape ifm_shape{1, 3, 244, 244};
  ConvolutionSpec spec{param.layer(1).convolution_param()};

  spec.ifm_shape(ifm_shape);

  // Check 'ker_shape'
  {
    auto expected = as_tensor_shape(net.layer_by_name("conv")->blobs().at(0)->shape());
    auto obtained = spec.ker_shape();

    ASSERT_EQ(expected, obtained);
  }

  // Check 'ofm_shape'
  {
    auto expected = as_tensor_shape(net.blob_by_name("conv")->shape());
    auto obtained = spec.ofm_shape();

    ASSERT_EQ(expected, obtained);
  }
}

namespace
{
// NOTE This example is derived from conv1_3x3_s2 layer in reference inception v3 layer
// clang-format off
const char *conv_2 = STRING(
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape: { dim: 1 dim: 3 dim: 299 dim: 299 }
  }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    bias_term: false
    num_output: 2
    stride: 2
    kernel_size: 3
  }
}
);
// clang-format on
} // namespace

TEST_F(ConvolutionSpecTest, conv_2)
{
  ::caffe::NetParameter param;

  ASSERT_TRUE(load(conv_2, param));

  ::caffe::Net<float> net{param};

  const tensor::Shape ifm_shape{1, 3, 299, 299};
  ConvolutionSpec spec{param.layer(1).convolution_param()};

  spec.ifm_shape(ifm_shape);

  // Check 'stride'
  ASSERT_EQ(spec.stride(0), 2);
  ASSERT_EQ(spec.stride(1), 2);

  // Check 'ofm_shape'
  {
    auto expected = as_tensor_shape(net.blob_by_name("conv")->shape());
    auto obtained = spec.ofm_shape();

    ASSERT_EQ(expected, obtained);
  }
}

namespace
{
// clang-format off
const char *conv_pad = STRING(
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape: { dim: 1 dim: 3 dim: 16 dim: 16 }
  }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    bias_term: false
    num_output: 2
    pad: 2
    kernel_size: 3
  }
}
);
// clang-format on
} // namespace

TEST_F(ConvolutionSpecTest, conv_pad)
{
  ::caffe::NetParameter param;

  ASSERT_TRUE(load(conv_pad, param));

  ::caffe::Net<float> net{param};

  const tensor::Shape ifm_shape{1, 3, 16, 16};
  ConvolutionSpec spec{param.layer(1).convolution_param()};

  spec.ifm_shape(ifm_shape);

  // Check 'pad'
  ASSERT_EQ(spec.pad(0), 2);
  ASSERT_EQ(spec.pad(1), 2);

  // Check 'ofm_shape'
  {
    auto expected = as_tensor_shape(net.blob_by_name("conv")->shape());
    auto obtained = spec.ofm_shape();

    ASSERT_EQ(expected, obtained);
  }
}

namespace
{
// clang-format off
const char *conv_ker_hw = STRING(
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape: { dim: 1 dim: 3 dim: 16 dim: 16 }
  }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    bias_term: false
    num_output: 2
    kernel_h: 3
    kernel_w: 1
  }
}
);
// clang-format on
} // namespace

TEST_F(ConvolutionSpecTest, conv_ker_hw)
{
  ::caffe::NetParameter param;

  ASSERT_TRUE(load(conv_ker_hw, param));

  ::caffe::Net<float> net{param};

  const tensor::Shape ifm_shape{1, 3, 16, 16};
  ConvolutionSpec spec{param.layer(1).convolution_param()};

  spec.ifm_shape(ifm_shape);

  // Check 'pad'
  ASSERT_EQ(spec.ker_dim(0), 3);
  ASSERT_EQ(spec.ker_dim(1), 1);

  // Check 'ofm_shape'
  {
    auto expected = as_tensor_shape(net.blob_by_name("conv")->shape());
    auto obtained = spec.ofm_shape();

    ASSERT_EQ(expected, obtained);
  }
}

namespace
{
// clang-format off
const char *dconv = STRING(
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape: { dim: 1 dim: 3 dim: 16 dim: 16 }
  }
}
layer {
  name: "conv"
  type: "Convolution"
  bottom: "data"
  top: "conv"
  convolution_param {
    bias_term: false
    num_output: 3
    kernel_size: 3
    group: 3
  }
}
);
// clang-format on
} // namespace

TEST_F(ConvolutionSpecTest, dconv)
{
  ::caffe::NetParameter param;

  ASSERT_TRUE(load(dconv, param));

  ::caffe::Net<float> net{param};

  const tensor::Shape ifm_shape{1, 3, 16, 16};
  ConvolutionSpec spec{param.layer(1).convolution_param()};

  spec.ifm_shape(ifm_shape);

  // Check 'ker_shape'
  {
    auto expected = as_tensor_shape(net.layer_by_name("conv")->blobs().at(0)->shape());
    auto obtained = spec.ker_shape();

    ASSERT_EQ(expected, obtained);
  }

  // Check 'ofm_shape'
  {
    auto expected = as_tensor_shape(net.blob_by_name("conv")->shape());
    auto obtained = spec.ofm_shape();

    ASSERT_EQ(expected, obtained);
  }
}
