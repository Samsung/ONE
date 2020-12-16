/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @brief This file includes various tests that shows how to encode TensorFlow models using loco.
 *
 * @note All the python examples below assume TensorFlow v1.13
 */
#include "loco.h"

#include <gtest/gtest.h>

#include <stdex/Memory.h>

using stdex::make_unique;

namespace
{

loco::Permutation<loco::Domain::Feature> make_NHWC_permutation(void)
{
  loco::Permutation<loco::Domain::Feature> NHWC;

  NHWC.axis(loco::FeatureAxis::Count) = 0;
  NHWC.axis(loco::FeatureAxis::Height) = 1;
  NHWC.axis(loco::FeatureAxis::Width) = 2;
  NHWC.axis(loco::FeatureAxis::Depth) = 3;

  return NHWC;
}

/**
 * @brief Create a HxWxIxO (or HxWxCxN) permutation which tf.nn.conv2d uses
 *
 * Reference: [tf.nn.conv2d](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
 * > Given an input tensor of shape [batch, in_height, in_width, in_channels] and a filter /
 * > kernel tensor of shape [filter_height, filter_width, in_channels, out_channels], ...
 *
 * NOTE "HWIO" is borrowed from TensorFlow Lite Converter
 *
 * https://github.com/tensorflow/tensorflow/blob/v1.13.1/tensorflow/lite/toco/model.h#L169
 */
loco::Permutation<loco::Domain::Filter> make_HWIO_permutation(void)
{
  loco::Permutation<loco::Domain::Filter> HWIO;

  HWIO.axis(loco::FilterAxis::Height) = 0; // H
  HWIO.axis(loco::FilterAxis::Width) = 1;  // W
  HWIO.axis(loco::FilterAxis::Depth) = 2;  // I, a.k.a. C
  HWIO.axis(loco::FilterAxis::Count) = 3;  // O, a.k.a. N

  return HWIO;
}

} // namespace

#if 0
>>> MaxPool_Float_000 testcase

MaxPool_Float_000 test guarantees that loco is expressive enough to encode the following example.

Python:
```
import tensorflow as tf
value = tf.placeholder(dtype=tf.float32, shape=[1, 16, 16, 2], name="value")
maxpool = tf.nn.max_pool(value, [1, 3, 3, 1], [1, 1, 1, 1], 'VALID', name="maxpool")
tf.get_default_graph().as_graph_def()
```

The above code produces the following TensorFlow GraphDef:

node {
  name: "value"
  op: "Placeholder"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim { size: 1 }
        dim { size: 16 }
        dim { size: 16 }
        dim { size: 2 }
      }
    }
  }
}
node {
  name: "maxpool"
  op: "MaxPool"
  input: "Placeholder"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
  attr {
    key: "data_format"
    value { s: "NHWC" }
  }
  attr {
    key: "ksize"
    value { list { i: 1 i: 3 i: 3 i: 1 } }
  }
  attr {
    key: "padding"
    value { s: "VALID" }
  }
  attr {
    key: "strides"
    value { list { i: 1 i: 1 i: 1 i: 1 } }
  }
}

Below test guarantees that loco is expressive enough to encode this example.
#endif
TEST(TensorFlowTest, MaxPool_Float_000)
{
  auto g = loco::make_graph();

  // The first "value" node corresponds to the following "Pull" node.
  //
  // %value = Pull(dtype: FLOAT32, shape: [1, 16, 16, 2])
  auto value = g->nodes()->create<loco::Pull>();

  value->dtype(loco::DataType::FLOAT32);
  value->shape({1, 16, 16, 2});

  // The next "maxpool" node corresponds to a sequence of the following loco nodes:
  //  - "FeatureEncode"
  //  - "MaxPool2D
  //  - "FeatureDecode"
  //
  // "maxpool.data_format" is 'NHWC' which corresponds to the following permutation
  //   Count  <-> 0
  //   Height <-> 1
  //   Width  <-> 2
  //   Depth  <-> 3
  loco::Permutation<loco::Domain::Feature> NHWC;

  NHWC.axis(loco::FeatureAxis::Count) = 0;
  NHWC.axis(loco::FeatureAxis::Height) = 1;
  NHWC.axis(loco::FeatureAxis::Width) = 2;
  NHWC.axis(loco::FeatureAxis::Depth) = 3;

  auto encoder = make_unique<loco::PermutingEncoder<loco::Domain::Feature>>();

  encoder->perm(NHWC);

  auto decoder = make_unique<loco::PermutingDecoder<loco::Domain::Feature>>();

  decoder->perm(NHWC);

  // %node_0 = FeatureEncode(%value, perm { Count = 0, Height = 1, Width = 2, Depth = 3 })
  auto node_0 = g->nodes()->create<loco::FeatureEncode>();

  node_0->input(value);
  node_0->encoder(std::move(encoder));

  // %node_1 = MaxPool(%node_0, window.H: 3, window.W: 3, stride.H: 1, stride.W : 1)
  auto node_1 = g->nodes()->create<loco::MaxPool2D>();

  node_1->ifm(node_0);

  // From "ksize" attributes
  node_1->window()->horizontal(3);
  node_1->window()->vertical(3);

  // From "strides" attributes
  node_1->stride()->horizontal(1);
  node_1->stride()->vertical(1);

  // %output = FeatureDecode(%node_1, perm { Count = 0, Height = 1, Width = 2, Depth = 3 })
  auto output = g->nodes()->create<loco::FeatureDecode>();

  output->input(node_1);
  output->decoder(std::move(decoder));

  // %push = Push(%output)
  auto push = g->nodes()->create<loco::Push>();

  push->from(output);

  //
  // Mark network-level input/output
  //
  auto input_0 = g->inputs()->create();
  loco::link(input_0, value);

  auto output_0 = g->outputs()->create();
  loco::link(output_0, push);

  // NOTE This example SHOULD BE valid.
  ASSERT_TRUE(loco::valid(g.get()));
}

#if 0
>>> Conv2D_Float_000 testcase

Conv2D_Float_000 test guarantees that loco is expressive enough to encode the following example.

Python:
```
import tensorflow as tf
inp = tf.placeholder(dtype=tf.float32, shape=[1, 16, 16, 2], name="inp")
ker = tf.constant(value=[1.0], dtype=tf.float32, shape=[7, 1, 2, 4], name="ker")
conv2d = tf.nn.conv2d(input=inp, filter=ker, strides=[1, 1, 1, 1], padding='VALID', name="conv2d")
tf.get_default_graph().as_graph_def()
```

TensorFlow GraphDef:
```
node {
  name: "inp"
  op: "Placeholder"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim { size: 1 }
        dim { size: 16 }
        dim { size: 16 }
        dim { size: 2 }
      }
    }
  }
}
node {
  name: "ker"
  op: "Const"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim { size: 7 }
          dim { size: 1 }
          dim { size: 2 }
          dim { size: 4 }
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "conv2d"
  op: "Conv2D"
  input: "inp"
  input: "ker"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
  attr {
    key: "data_format"
    value { s: "NHWC" }
  }
  attr {
    key: "dilations"
    value { list { i: 1 i: 1 i: 1 i: 1 } }
  }
  attr {
    key: "padding"
    value { s: "VALID" }
  }
  attr {
    key: "strides"
    value { list { i: 1 i: 1 i: 1 i: 1 } }
  }
}
```
#endif
TEST(TensorFlowTest, Conv2D_Float_000)
{
  auto g = loco::make_graph();

  // The first "inp" node corresponds to "Pull"
  auto inp = g->nodes()->create<loco::Pull>();
  {
    inp->dtype(loco::DataType::FLOAT32);
    inp->shape({1, 16, 16, 2});
  }

  // The seoncd "ker" node corresponds to "ConstGen"
  auto ker = g->nodes()->create<loco::ConstGen>();
  {
    ker->dtype(loco::DataType::FLOAT32);
    // 'I' denotes IFM DEPTH, and 'O' denotes OFM DEPTH
    ker->shape({7 /*H*/, 1 /*W*/, 2 /*I*/, 3 /*O*/});
    ker->size<loco::DataType::FLOAT32>(7 * 1 * 2 * 3);
    for (uint32_t n = 0; n < 7 * 1 * 2 * 3; ++n)
    {
      // NOTE TensorFlow uses the last value to fill unspecified region
      ker->at<loco::DataType::FLOAT32>(n) = 1.0f;
    }
  }

  // The next "conv2d" node is decomposed into the following loco nodes
  //  - "FeatureEncode"
  //  - "FilterEncode"
  //  - "Conv2D"
  //  - "FeatureDecode"
  auto encoded_ifm = g->nodes()->create<loco::FeatureEncode>();
  {
    // From "conv2d.data_format" attribute
    auto encoder = make_unique<loco::PermutingEncoder<loco::Domain::Feature>>();
    encoder->perm(make_NHWC_permutation());

    encoded_ifm->input(inp);
    encoded_ifm->encoder(std::move(encoder));
  }

  auto encoded_ker = g->nodes()->create<loco::FilterEncode>();
  {
    // From "tf.nn.conv2d" specification
    auto encoder = make_unique<loco::PermutingEncoder<loco::Domain::Filter>>();
    encoder->perm(make_HWIO_permutation());

    encoded_ker->input(ker);
    encoded_ker->encoder(std::move(encoder));
  }

  auto conv2d = g->nodes()->create<loco::Conv2D>();
  {
    conv2d->ifm(encoded_ifm);
    conv2d->ker(encoded_ker);

    // From "stride" attribute
    conv2d->stride()->horizontal(1);
    conv2d->stride()->vertical(1);
  }

  // "decoded_ofm" corresponds to the output of "conv2d" node.
  auto decoded_ofm = g->nodes()->create<loco::FeatureDecode>();
  {
    // From "conv2d.data_format" attribute
    auto decoder = make_unique<loco::PermutingDecoder<loco::Domain::Feature>>();
    decoder->perm(make_NHWC_permutation());

    decoded_ofm->input(conv2d);
    decoded_ofm->decoder(std::move(decoder));
  }

  // Makr "conv2d" as a network-level output with Push
  auto push = g->nodes()->create<loco::Push>();
  {
    push->from(decoded_ofm);
  }

  //
  // Mark network-level input/output
  //
  auto input_0 = g->inputs()->create();
  loco::link(input_0, inp);

  auto output_0 = g->outputs()->create();
  loco::link(output_0, push);

  // NOTE This example SHOULD BE valid.
  ASSERT_TRUE(loco::valid(g.get()));
}
