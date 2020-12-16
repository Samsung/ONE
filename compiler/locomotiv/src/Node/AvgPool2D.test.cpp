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

using nncc::core::ADT::tensor::Shape;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::make_overlay;

void run_test(const float *ifm, const float *expected_ofm, const Shape &ifm_shape,
              const Shape &ofm_shape, const uint32_t window_v, const uint32_t window_h,
              const uint32_t stride_v, const uint32_t stride_h, const uint32_t pad_top,
              const uint32_t pad_bottom, const uint32_t pad_left, const uint32_t pad_right)
{
  // Let's make FeatureEncode-AvgPool2D graph
  auto g = loco::make_graph();
  auto enc = g->nodes()->create<loco::FeatureEncode>();

  // Fill output data of FeatureEncode from ifm
  auto enc_buf = make_buffer<float, LexicalLayout>(ifm_shape);

  auto ifm_overlay = make_overlay<float, LexicalLayout>(ifm_shape, const_cast<float *>(ifm));
  for (nncc::core::ADT::tensor::IndexEnumerator e{ifm_shape}; e.valid(); e.advance())
  {
    const auto &ind = e.current();
    enc_buf.at(ind) = ifm_overlay.at(ind);
  }

  auto enc_data = locomotiv::make_data(enc_buf);
  locomotiv::annot_data(enc, std::move(enc_data));
  locomotiv::annot_domain(enc, loco::Domain::Feature);

  // build TF AvgPool2D
  auto avgpool2d = g->nodes()->create<loco::AvgPool2D>();
  avgpool2d->ifm(enc);
  avgpool2d->convention(loco::AvgPool2D::Convention::Valid);
  avgpool2d->window()->vertical(window_v);
  avgpool2d->window()->horizontal(window_h);
  avgpool2d->stride()->vertical(stride_v);
  avgpool2d->stride()->horizontal(stride_h);
  avgpool2d->pad()->top(pad_top);
  avgpool2d->pad()->bottom(pad_bottom);
  avgpool2d->pad()->left(pad_left);
  avgpool2d->pad()->right(pad_right);

  // run interpreter
  locomotiv::NodeExecution::get().run(avgpool2d);

  // get result of calculation
  auto avgpool2d_data = locomotiv::annot_data(avgpool2d);

  // check the result
  ASSERT_NE(avgpool2d_data, nullptr);
  ASSERT_TRUE(avgpool2d_data->dtype() == loco::DataType::FLOAT32);
  ASSERT_TRUE(*(avgpool2d_data->shape()) == ofm_shape);

  auto ofm_overlay =
    make_overlay<float, LexicalLayout>(ofm_shape, const_cast<float *>(expected_ofm));
  for (nncc::core::ADT::tensor::IndexEnumerator e{ofm_shape}; e.valid(); e.advance())
  {
    const auto &ind = e.current();
    ASSERT_FLOAT_EQ(ofm_overlay.at(ind), avgpool2d_data->as_f32_bufptr()->at(ind));
  }

  ASSERT_EQ(loco::Domain::Feature, locomotiv::annot_domain(avgpool2d));
}

} // namespace

// clang-format off
/* ifm and ofm are from the code below:
import tensorflow as tf

value = tf.constant([[[[-0.281157], [-1.0601869], [-0.622261],  [-1.1777412]],
                      [[1.4411974], [0.01408334], [0.06958964], [-0.08663343]],
                      [[1.3424183], [-0.89015573], [0.2520576], [0.04843695]],
                      [[-1.6668711], [-0.02187406], [1.9362065], [1.3341236]]]])
avgpool = tf.nn.avg_pool(value, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding= 'VALID',
                         data_format="NHWC")
with tf.Session() as sess:
    print(sess.run(avgpool))
*/
TEST(NodeExecution_AvgPool2D, f32_1x4x4x1_calculation)
{
  using nncc::core::ADT::tensor::Shape;

  const float ifm[] =
  {
    -0.281157,  -1.0601869,  -0.622261,   -1.1777412,
     1.4411974,  0.01408334,  0.06958964, -0.08663343,
     1.3424183, -0.89015573,  0.2520576,   0.04843695,
    -1.6668711, -0.02187406,  1.9362065,   1.3341236
  };

  const float ofm[] =
  {
     0.02848421, -0.45426148,
    -0.30912063,  0.89270616
  };

  run_test(ifm, ofm,
           Shape{1, 4, 4, 1}, Shape{1, 2, 2, 1}, // input shape , output shape
           2, 2,       // kernel
           2, 2,       // stride
           0, 0, 0, 0  // padding
  );
}
// clang-format on

// clang-format off
/* ifm and ofm are from the code below:
import tensorflow as tf

value = tf.constant([[[[-0.281157], [-1.0601869], [-0.622261]],
                      [[1.4411974], [0.01408334], [0.06958964]],
                      [[1.3424183], [-0.89015573], [0.2520576]]]])
avgpool = tf.nn.avg_pool(value, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding= 'SAME',
                         data_format="NHWC")
with tf.Session() as sess:
    print(sess.run(avgpool))
*/
TEST(NodeExecution_AvgPool2D, f32_1x3x3x1_calculation)
{
  using nncc::core::ADT::tensor::Shape;

  const float ifm[] =
  {
    -0.281157,  -1.0601869,  -0.622261,
     1.4411974,  0.01408334,  0.06958964,
     1.3424183, -0.89015573,  0.2520576
  };

  const float ofm[] =
  {
     0.02848421, -0.39969373, -0.2763357,
     0.4768858,  -0.13860628,  0.16082363,
     0.22613129, -0.31904906,  0.2520576
  };

  run_test(ifm, ofm,
           Shape{1, 3, 3, 1}, Shape{1, 3, 3, 1}, // input shape , output shape
           2, 2,       // kernel
           1, 1,       // stride
           0, 1, 0, 1  // padding
  );
}
// clang-format on
