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

  // build TransposedConv2D
  auto tr_conv2d = g->nodes()->create<loco::TransposedConv2D>();
  tr_conv2d->ifm(ifm_enc);
  tr_conv2d->ker(ker_enc);
  tr_conv2d->stride()->vertical(stride_v);
  tr_conv2d->stride()->horizontal(stride_h);
  tr_conv2d->pad()->top(pad_top);
  tr_conv2d->pad()->bottom(pad_bottom);
  tr_conv2d->pad()->left(pad_left);
  tr_conv2d->pad()->right(pad_right);

  // run interpreter
  locomotiv::NodeExecution::get().run(tr_conv2d);

  // get result of calculation
  auto conv2d_result = locomotiv::annot_data(tr_conv2d);

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

  ASSERT_EQ(loco::Domain::Feature, locomotiv::annot_domain(tr_conv2d));
}

} // namespace

// clang-format off
/*
ifm = tf.constant(1.1, shape = [1, 2, 2, 4])
ker = tf.constant(2.2, shape = [3, 3, 2, 4])
tr_conv = tf.nn.conv2d_transpose(ifm, ker, output_shape = (1, 5, 5, 2), strides = [1, 2, 2, 1], padding = "VALID")

with tf.Session() as session:
  tr_conv_data = session.run(tr_conv)
 */
TEST(NodeExecution_TransposedConv2D, f32)
{
  using nncc::core::ADT::tensor::Shape;

  float ifm[1 * 2 * 2 * 4];
  for (int n = 0; n < 1 * 2 * 2 * 4; n++)
    ifm[n] = 1.1;

  float ker[2 * 3 * 3 * 4]; // NHWC 
  for (int n = 0; n < 2 * 3 * 3 * 4; n++)
    ker[n] = 2.2;

  float ofm[1 * 5 * 5 * 2] = {9.68,  9.68,  9.68,  9.68,  19.36, 19.36, 9.68,  9.68,  9.68,  9.68,
                              9.68,  9.68,  9.68,  9.68,  19.36, 19.36, 9.68,  9.68,  9.68,  9.68,
                              19.36, 19.36, 19.36, 19.36, 38.72, 38.72, 19.36, 19.36, 19.36, 19.36,
                              9.68,  9.68,  9.68,  9.68,  19.36, 19.36, 9.68,  9.68,  9.68,  9.68,
                              9.68,  9.68,  9.68,  9.68,  19.36, 19.36, 9.68,  9.68,  9.68,  9.68};

  run_test(ifm, ker, ofm,
           Shape{1, 2, 2, 4}, Shape{2, 3, 3, 4}, Shape{1, 5, 5, 2}, // shapes of ifm, ker, ofm
           2, 2 // stride
           );
}
// clang-format on
