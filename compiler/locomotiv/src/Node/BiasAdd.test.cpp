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

#include "NodeExecution.h"

#include "locomotiv/NodeData.h"
#include "NodeDataImpl.h"
#include "NodeDomain.h"

#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include <nncc/core/ADT/tensor/Index.h>
#include <nncc/core/ADT/tensor/IndexEnumerator.h>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

/*
test case generated from the following:

  inp = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                    shape=[1, 3, 3, 2], dtype=tf.float32)
  bias = tf.constant([1.1, 2.1], shape=[2], dtype=tf.float32)
  out = tf.nn.bias_add(inp, bias)

  with tf.Session() as sess:
      print(sess.run(out))
 */

TEST(NodeExecution_TensorBiasAdd, f32)
{
  float in_val[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  float bias_val[] = {1.1, 2.1};
  float out_val[] = {2.1,  4.1,  4.1,  6.1,  6.1,  8.1,  8.1,  10.1, 10.1,
                     12.1, 12.1, 14.1, 14.1, 16.1, 16.1, 18.1, 18.1, 20.1};

  // make BiasAdd(Pull, Const)
  auto g = loco::make_graph();
  Shape input_shape{1, 3, 3, 2}; // NHWC

  auto inp = g->nodes()->create<loco::Pull>();
  {
    inp->dtype(loco::DataType::FLOAT32);
    inp->shape({1, 3, 3, 2});
  }

  auto bias = g->nodes()->create<loco::BiasEncode>();
  {
    // nothing to do
  }

  auto bias_add = g->nodes()->create<loco::BiasAdd<loco::Domain::Tensor>>();
  {
    bias_add->value(inp);
    bias_add->bias(bias);
    bias_add->axis(3); // axis(3) means C in NHWC
  }

  // Make and assign data to pull node
  auto inp_buf = make_buffer<float, LexicalLayout>(input_shape);
  {
    int n = 0;
    for (IndexEnumerator e{inp_buf.shape()}; e.valid(); e.advance())
    {
      inp_buf.at(e.current()) = in_val[n++];
    }
  }

  auto bias_buf = make_buffer<float, LexicalLayout>(Shape{2});
  {
    int n = 0;
    for (IndexEnumerator e{bias_buf.shape()}; e.valid(); e.advance())
    {
      bias_buf.at(e.current()) = bias_val[n++];
    }
  }

  auto inp_data = locomotiv::make_data(inp_buf);
  locomotiv::annot_data(inp, std::move(inp_data));
  locomotiv::annot_domain(inp, loco::Domain::Tensor);

  auto bias_data = locomotiv::make_data(bias_buf);
  locomotiv::annot_data(bias, std::move(bias_data));
  locomotiv::annot_domain(bias, loco::Domain::Bias);

  locomotiv::NodeExecution::get().run(bias_add);

  auto bias_add_data = locomotiv::annot_data(bias_add);

  // comparing the result
  ASSERT_NE(bias_add_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, bias_add_data->dtype());
  ASSERT_EQ(Shape({1, 3, 3, 2}), *(bias_add_data->shape()));

  uint32_t n = 0;
  for (IndexEnumerator e{*(bias_add_data->shape())}; e.valid(); e.advance())
  {
    ASSERT_FLOAT_EQ(out_val[n++], bias_add_data->as_f32_bufptr()->at(e.current()));
  }

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(bias_add));
}

/*
test case generated from the following:

  inp = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                    shape=[1, 3, 3, 2], dtype=tf.float32)
  bias = tf.constant([1.1, 2.1], shape=[2], dtype=tf.float32)
  out = tf.nn.bias_add(inp, bias)

  with tf.Session() as sess:
      print(sess.run(out))
 */

TEST(NodeExecution_FeatureBiasAdd, f32)
{
  float in_val[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  float bias_val[] = {1.1, 2.1};
  float out_val[] = {2.1,  4.1,  4.1,  6.1,  6.1,  8.1,  8.1,  10.1, 10.1,
                     12.1, 12.1, 14.1, 14.1, 16.1, 16.1, 18.1, 18.1, 20.1};

  // make FeatureBiasAdd(FeatureEncode, BiasEncode)
  auto g = loco::make_graph();
  Shape input_shape{1, 3, 3, 2}; // NHWC

  auto feature_encode = g->nodes()->create<loco::FeatureEncode>();
  {
    // setting values is ignored for testing
  }

  auto bias = g->nodes()->create<loco::BiasEncode>();
  {
    // nothing to do
  }

  auto feature_bias_add = g->nodes()->create<loco::BiasAdd<loco::Domain::Feature>>();
  {
    feature_bias_add->value(feature_encode);
    feature_bias_add->bias(bias);
  }

  // Make and assign data to pull node
  auto inp_buf = make_buffer<float, LexicalLayout>(input_shape);
  {
    int n = 0;
    for (IndexEnumerator e{inp_buf.shape()}; e.valid(); e.advance())
    {
      inp_buf.at(e.current()) = in_val[n++];
    }
  }

  auto bias_buf = make_buffer<float, LexicalLayout>(Shape{2});
  {
    int n = 0;
    for (IndexEnumerator e{bias_buf.shape()}; e.valid(); e.advance())
    {
      bias_buf.at(e.current()) = bias_val[n++];
    }
  }

  auto inp_data = locomotiv::make_data(inp_buf);
  locomotiv::annot_data(feature_encode, std::move(inp_data));
  locomotiv::annot_domain(feature_encode, loco::Domain::Feature);

  auto bias_data = locomotiv::make_data(bias_buf);
  locomotiv::annot_data(bias, std::move(bias_data));
  locomotiv::annot_domain(bias, loco::Domain::Bias);

  locomotiv::NodeExecution::get().run(feature_bias_add);

  auto bias_add_data = locomotiv::annot_data(feature_bias_add);

  // comparing the result
  ASSERT_NE(bias_add_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, bias_add_data->dtype());
  ASSERT_EQ(Shape({1, 3, 3, 2}), *(bias_add_data->shape()));

  uint32_t n = 0;
  for (IndexEnumerator e{*(bias_add_data->shape())}; e.valid(); e.advance())
  {
    ASSERT_FLOAT_EQ(out_val[n++], bias_add_data->as_f32_bufptr()->at(e.current()));
  }

  ASSERT_EQ(loco::Domain::Feature, locomotiv::annot_domain(feature_bias_add));
}
