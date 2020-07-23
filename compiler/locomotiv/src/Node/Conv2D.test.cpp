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
#include <nncc/core/ADT/tensor/Overlay.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>
#include "nncc/core/ADT/tensor/IndexEnumerator.h"

#include <gtest/gtest.h>

namespace
{
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::make_overlay;
using nncc::core::ADT::tensor::Shape;

void run_test(const float *ifm, const float *ker, const float *expected_ofm, const Shape &ifm_shape,
              const Shape ker_shape, const Shape ofm_shape, const uint32_t stride_v,
              const uint32_t stride_h, const uint32_t pad_top = 0, const uint32_t pad_bottom = 0,
              const uint32_t pad_left = 0, const uint32_t pad_right = 0)
{
  auto g = loco::make_graph();

  // Fill output data of FeatureEncode from ifm
  auto ifm_enc = g->nodes()->create<loco::FeatureEncode>();
  {
    auto ifm_enc_buf = make_buffer<float, LexicalLayout>(ifm_shape);
    auto ifm_overlay = make_overlay<float, LexicalLayout>(ifm_shape, const_cast<float *>(ifm));
    for (nncc::core::ADT::tensor::IndexEnumerator e{ifm_shape}; e.valid(); e.advance())
    {
      const auto &ind = e.current();
      ifm_enc_buf.at(ind) = ifm_overlay.at(ind);
    }

    auto enc_data = locomotiv::make_data(ifm_enc_buf);
    locomotiv::annot_data(ifm_enc, std::move(enc_data));
    locomotiv::annot_domain(ifm_enc, loco::Domain::Feature);
  }

  // Fill output data of FilterEncode from ker
  auto ker_enc = g->nodes()->create<loco::FilterEncode>();
  {
    auto ker_enc_buf = make_buffer<float, LexicalLayout>(ker_shape);
    auto ker_overlay = make_overlay<float, LexicalLayout>(ker_shape, const_cast<float *>(ker));
    for (nncc::core::ADT::tensor::IndexEnumerator e{ker_shape}; e.valid(); e.advance())
    {
      const auto &ind = e.current();
      ker_enc_buf.at(ind) = ker_overlay.at(ind);
    }

    auto enc_data = locomotiv::make_data(ker_enc_buf);
    locomotiv::annot_data(ker_enc, std::move(enc_data));
    locomotiv::annot_domain(ker_enc, loco::Domain::Filter);
  }

  // build Conv2D
  auto conv2d = g->nodes()->create<loco::Conv2D>();
  conv2d->ifm(ifm_enc);
  conv2d->ker(ker_enc);
  conv2d->stride()->vertical(stride_v);
  conv2d->stride()->horizontal(stride_h);
  conv2d->pad()->top(pad_top);
  conv2d->pad()->bottom(pad_bottom);
  conv2d->pad()->left(pad_left);
  conv2d->pad()->right(pad_right);

  // run interpreter
  locomotiv::NodeExecution::get().run(conv2d);

  // get result of calculation
  auto conv2d_result = locomotiv::annot_data(conv2d);

  // check the result
  ASSERT_NE(conv2d_result, nullptr);
  ASSERT_TRUE(conv2d_result->dtype() == loco::DataType::FLOAT32);
  ASSERT_TRUE(*(conv2d_result->shape()) == ofm_shape);

  auto ofm_overlay =
      make_overlay<float, LexicalLayout>(ofm_shape, const_cast<float *>(expected_ofm));
  for (nncc::core::ADT::tensor::IndexEnumerator e{ofm_shape}; e.valid(); e.advance())
  {
    const auto &ind = e.current();
    ASSERT_FLOAT_EQ(ofm_overlay.at(ind), conv2d_result->as_f32_bufptr()->at(ind));
  }

  ASSERT_EQ(loco::Domain::Feature, locomotiv::annot_domain(conv2d));
}

} // namespace

// clang-format off
/* ifm and ofm are from the code below:

ifm = tf.random_normal([1, 5, 5, 1], stddev=1)
ker = tf.random_normal([3, 3, 1, 1], stddev=1)
out = tf.nn.conv2d(ifm, ker, strides = [1, 2, 2, 1], padding= 'VALID')

with tf.Session() as sess:
    print(sess.run(out))
*/
TEST(NodeExecution_Conv2D, f32_1x5x5x1_calculation)
{
  using nncc::core::ADT::tensor::Shape;

  const float ifm[] =
  {
    -0.48850584,  1.4292705,  -1.3424522, -0.7441476,  -1.8964586,
     1.7021934,  -0.39246717,  0.6248314,  0.12724274,  1.3915083,
     0.382255,    0.7725081,   0.9171561, -1.1847119,   0.61858755,
     1.1530193,  -0.476239,   -0.9038663, -0.48764458,  0.339963,
     2.2817912,  -0.8464133,  -1.0598192,  0.8361126,   1.2344601
  };

  const float ker[] =
  {
    -0.0830195,  0.21088193, -0.11781317,
     0.07755677, 1.6337638,   1.0792778,
    -1.6922939, -1.5437212,   0.96667504
  };

  const float ofm[] =
  {
    -0.28752697, 2.8108592,
    -5.220376  , 0.7973861
  };

  run_test(ifm, ker, ofm,
           Shape{1, 5, 5, 1}, Shape{1, 3, 3, 1}, Shape{1, 2, 2, 1}, // shapes of input, ker, output
           2, 2  // stride
  );
}

TEST(NodeExecution_Conv2D, f32_multiple_channel)
{
  // testing channel != 1, stride = [1,1]
  using nncc::core::ADT::tensor::Shape;

  float ifm[1*5*5*3];
  for (int n = 0; n < 5*5*3; n++) ifm[n] = 2.2;

  float ker[2*2*2*3]; // nhwc
  for (int n = 0; n < 2*2*2*3; n++) ker[n] = 1.1;

  float ofm[1*4*4*2];
  for (int n = 0; n < 1*4*4*2; n++) ofm[n] = 29.04;

  run_test(ifm, ker, ofm,
           Shape{1, 5, 5, 3}, Shape{2, 2, 2, 3}, Shape{1, 4, 4, 2}, // shapes of input, ker, output
           1, 1  // stride
  );
}

/* ifm and ofm are from the code below:
tensorflow version : 1.12.0

import tensorflow as tf

ifm = tf.constant([-1.3653529,  0.4160791,  0.5059157,  0.7649683,  0.39364856,
        -1.0164733,  1.506766,  -1.1413091,  1.2766701, -0.9253511,
        1.3570246,  0.32089928,  -0.9898171,  1.983792,  -0.3423274,
        -1.1901658,  1.2288222,  -0.47401968,  -0.01369802,  0.4136331,
        0.06960588,  -0.16537654,  -0.65015996,  -0.555224,  0.7140603
], shape=[1, 5, 5, 1])

ker = tf.constant([2.3490515,  -0.4572366,  0.05790535,
        0.3672005,  0.52679914,  0.74607974,
        -1.7211207,  1.1174419,  -0.59663385
], shape=[3, 3, 1, 1])

ofm = tf.nn.conv2d(ifm, ker, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    print(sess.run(ofm))
*/
TEST(NodeExecution_Conv2D, with_padding)
{
  using nncc::core::ADT::tensor::Shape;

  const float ifm[] =
  {
    -1.3653529,  0.4160791,  0.5059157,  0.7649683,  0.39364856,
    -1.0164733,  1.506766,  -1.1413091,  1.2766701, -0.9253511,
    1.3570246,  0.32089928,  -0.9898171,  1.983792,  -0.3423274,
    -1.1901658,  1.2288222,  -0.47401968,  -0.01369802,  0.4136331,
    0.06960588,  -0.16537654,  -0.65015996,  -0.555224,  0.7140603 
  };

  const float ker[] =
  {
    2.3490515,  -0.4572366,  0.05790535,
    0.3672005,  0.52679914,  0.74607974,
    -1.7211207,  1.1174419,  -0.59663385
  };

  const float ofm[] =
  {
    -2.443676,  4.2094254,  -3.6403496,  4.8254814,  -2.743059,
    2.5620093,  -5.185688,  -1.1470609,  4.54913,  -2.1985974,
    -0.5567835,  0.49045527,  2.5752437,  -2.3383713,  4.455967,
    -0.13562866,  2.9236434,  1.4019353,  -3.0521483,  6.782954,
    0.5286269,  -3.9317036,  2.285041,  -1.0817666,  -0.04901773
  };

  run_test(ifm, ker, ofm,
           Shape{1, 5, 5, 1}, Shape{1, 3, 3, 1}, Shape{1, 5, 5, 1}, // shapes of input, ker, output
           1, 1,  // stride
           1, 1, 1, 1  // padding
  );
}
// clang-format on
