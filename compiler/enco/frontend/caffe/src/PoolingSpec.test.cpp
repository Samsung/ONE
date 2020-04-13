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

#include "PoolingSpec.h"
#include "Importer.h"

#include <nncc/core/ADT/tensor/Shape.h>

#include <caffe/net.hpp>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <sstream>
#include <stdexcept>

#include <gtest/gtest.h>

using namespace nncc::core::ADT;

#define STRING(content) #content

bool from_txt(const std::string &txt, ::caffe::PoolingParameter &out)
{
  std::stringstream ss{txt};
  return from_txt(ss, out);
}

namespace
{

class SequentialBuilder
{
public:
  SequentialBuilder(::caffe::NetParameter *net) : _net{net}
  {
    // DO NOTHING
  }

public:
  bool addLayer(const std::string &prototxt)
  {
    auto layer = _net->add_layer();
    std::stringstream ss{prototxt};
    ::google::protobuf::io::IstreamInputStream iis{&ss};
    return google::protobuf::TextFormat::Parse(&iis, layer);
  }

  bool addInputLayer(const tensor::Shape &shape)
  {
    auto param = new ::caffe::InputParameter;
    {
      auto s = param->add_shape();
      for (uint32_t n = 0; n < shape.rank(); ++n)
      {
        s->add_dim(shape.dim(n));
      }
    }

    auto layer = _net->add_layer();

    layer->set_name("data");
    layer->set_type("Input");
    layer->add_top("data");
    layer->set_allocated_input_param(param);

    return true;
  }

private:
  ::caffe::NetParameter *_net;
};

} // namespace

namespace
{

class PoolingSpecTest : public ::testing::Test
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
};
} // namespace

TEST_F(PoolingSpecTest, ifm_shape)
{
  ::caffe::PoolingParameter param;
  PoolingSpec spec{param};

  const tensor::Shape ifm_shape{1, 3, 244, 244};

  spec.ifm_shape(ifm_shape);

  ASSERT_EQ(spec.ifm_shape(), ifm_shape);
}

namespace
{
} // namespace

TEST_F(PoolingSpecTest, kernel_size_same_for_all)
{
  const tensor::Shape ifm_shape{1, 3, 16, 16};

  ::caffe::NetParameter param;
  {
    SequentialBuilder builder{&param};

    builder.addInputLayer(ifm_shape);

    // clang-format off
    const char *prototxt = STRING(
      name : "pool"
      type : "Pooling"
      bottom : "data"
      top : "pool"
      pooling_param { kernel_size : 3 }
    );
    // clang-format on

    builder.addLayer(prototxt);
  }

  ::caffe::Net<float> net{param};

  PoolingSpec spec{param.layer(1).pooling_param()};

  spec.ifm_shape(ifm_shape);

  ASSERT_EQ(spec.window_height(), 3);
  ASSERT_EQ(spec.window_width(), 3);

  // Check 'ofm_shape'
  {
    auto expected = as_tensor_shape(net.blob_by_name("pool")->shape());
    auto obtained = spec.ofm_shape();

    ASSERT_EQ(expected, obtained);
  }
}

TEST_F(PoolingSpecTest, pad_for_all)
{
  const tensor::Shape ifm_shape{1, 3, 15, 15};

  ::caffe::NetParameter param;
  {
    SequentialBuilder builder{&param};

    builder.addInputLayer(ifm_shape);

    // clang-format off
    const char *prototxt = STRING(
      name : "pool"
      type : "Pooling"
      bottom : "data"
      top : "pool"
      pooling_param {
        pool: MAX
        kernel_size : 3
        pad: 2
      }
    );
    // clang-format on

    builder.addLayer(prototxt);
  }

  ::caffe::Net<float> net{param};

  PoolingSpec spec{param.layer(1).pooling_param()};

  spec.ifm_shape(ifm_shape);

  ASSERT_EQ(spec.vertical_pad(), 2);
  ASSERT_EQ(spec.horizontal_pad(), 2);

  // Check 'ofm_shape'
  {
    auto expected = as_tensor_shape(net.blob_by_name("pool")->shape());
    auto obtained = spec.ofm_shape();

    ASSERT_EQ(expected, obtained);
  }
}

TEST_F(PoolingSpecTest, stride_for_all)
{
  const tensor::Shape ifm_shape{1, 3, 15, 15};

  ::caffe::NetParameter param;
  {
    SequentialBuilder builder{&param};

    builder.addInputLayer(ifm_shape);

    // clang-format off
    const char *prototxt = STRING(
      name : "pool"
      type : "Pooling"
      bottom : "data"
      top : "pool"
      pooling_param {
        pool: MAX
        kernel_size : 3
        stride: 2
      }
    );
    // clang-format on

    builder.addLayer(prototxt);
  }

  ::caffe::Net<float> net{param};

  PoolingSpec spec{param.layer(1).pooling_param()};

  spec.ifm_shape(ifm_shape);

  ASSERT_EQ(spec.vertical_stride(), 2);
  ASSERT_EQ(spec.horizontal_stride(), 2);

  // Check 'ofm_shape'
  {
    auto expected = as_tensor_shape(net.blob_by_name("pool")->shape());
    auto obtained = spec.ofm_shape();

    ASSERT_EQ(expected, obtained);
  }
}

TEST_F(PoolingSpecTest, method_none)
{
  const char *prototxt = "";

  ::caffe::PoolingParameter param;
  from_txt(prototxt, param);

  PoolingSpec spec{param};

  ASSERT_EQ(spec.method(), PoolingMethod::Max);
}

TEST_F(PoolingSpecTest, method_max)
{
  const char *prototxt = "pool: MAX";

  ::caffe::PoolingParameter param;
  from_txt(prototxt, param);

  PoolingSpec spec{param};

  ASSERT_EQ(spec.method(), PoolingMethod::Max);
}

TEST_F(PoolingSpecTest, method_avg)
{
  const char *prototxt = "pool: AVE";

  ::caffe::PoolingParameter param;
  from_txt(prototxt, param);

  PoolingSpec spec{param};

  ASSERT_EQ(spec.method(), PoolingMethod::Avg);
}
