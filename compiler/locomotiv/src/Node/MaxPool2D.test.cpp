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
  // Let's make FeatureEncode-MaxPool2D graph
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

  // build MaxPool2D
  auto maxpool2d = g->nodes()->create<loco::MaxPool2D>();
  maxpool2d->ifm(enc);
  maxpool2d->window()->vertical(window_v);
  maxpool2d->window()->horizontal(window_h);
  maxpool2d->stride()->vertical(stride_v);
  maxpool2d->stride()->horizontal(stride_h);
  maxpool2d->pad()->top(pad_top);
  maxpool2d->pad()->bottom(pad_bottom);
  maxpool2d->pad()->left(pad_left);
  maxpool2d->pad()->right(pad_right);

  // run interpreter
  locomotiv::NodeExecution::get().run(maxpool2d);

  // get result of calculation
  auto maxpool2d_data = locomotiv::annot_data(maxpool2d);

  // check the result
  ASSERT_NE(maxpool2d_data, nullptr);
  ASSERT_TRUE(maxpool2d_data->dtype() == loco::DataType::FLOAT32);
  ASSERT_TRUE(*(maxpool2d_data->shape()) == ofm_shape);

  auto ofm_overlay =
    make_overlay<float, LexicalLayout>(ofm_shape, const_cast<float *>(expected_ofm));
  for (nncc::core::ADT::tensor::IndexEnumerator e{ofm_shape}; e.valid(); e.advance())
  {
    const auto &ind = e.current();
    ASSERT_FLOAT_EQ(ofm_overlay.at(ind), maxpool2d_data->as_f32_bufptr()->at(ind));
  }

  ASSERT_EQ(loco::Domain::Feature, locomotiv::annot_domain(maxpool2d));
}

} // namespace

// clang-format off
/* ifm and ofm are from the code below:

  value = tf.random_normal([1, 3, 3, 1], stddev=1)
  maxpool = tf.nn.max_pool(value, ksize = [1, 2, 2, 1], strides = [1, 1, 1, 1], padding= 'VALID',
                          data_format="NHWC")
  with tf.Session() as sess:
      print(sess.run(maxpool))
*/

TEST(NodeExecution_MaxPool2D, f32_1x3x3x1_calculation)
{
  using nncc::core::ADT::tensor::Shape;

  const float ifm[] =
  {
    -1.5510627,   0.3653609,    1.9002001,
    -0.15861237,  -0.32944828,  1.2053918,
    0.50054574,  -0.8533826,   0.131492,
  };

  const float ofm[] =
  {
    0.3653609, 1.9002001,
    0.50054574, 1.2053918
  };

  run_test(ifm, ofm,
           Shape{1, 3, 3, 1}, Shape{1, 2, 2, 1}, // input shape , output shape
           2, 2,       // kernel
           1, 1,       // stride
           0, 0, 0, 0  // padding
  );
}

TEST(NodeExecution_MaxPool2D, with_padding)
{
  using nncc::core::ADT::tensor::Shape;

  const float ifm[] =
  {
     1,  2,  3,  4,  5,
     6,  7,  8,  9, 10,
    11, 12, 13, 14, 15,
    16, 17, 18, 19, 20,
    21, 22, 23, 24, 25
  };

  const float ofm[] =
  {
     7,  9, 10,
    17, 19, 20,
    22, 24, 25
  };

  run_test(ifm, ofm,
           Shape{1, 5, 5, 1}, Shape{1, 3, 3, 1}, // input shape , output shape
           3, 3,       // kernel
           2, 2,       // stride
           1, 1, 1, 1  // padding - this mimics SAME padding
  );
}
// clang-format on
