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

  // Fill output data of DepthwiseFilterEncode from ker
  auto ker_enc = g->nodes()->create<loco::DepthwiseFilterEncode>();
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
    locomotiv::annot_domain(ker_enc, loco::Domain::DepthwiseFilter);
  }

  // build DepthwiseConv2D
  auto dw_conv2d = g->nodes()->create<loco::DepthwiseConv2D>();
  dw_conv2d->ifm(ifm_enc);
  dw_conv2d->ker(ker_enc);
  dw_conv2d->stride()->vertical(stride_v);
  dw_conv2d->stride()->horizontal(stride_h);
  dw_conv2d->pad()->top(pad_top);
  dw_conv2d->pad()->bottom(pad_bottom);
  dw_conv2d->pad()->left(pad_left);
  dw_conv2d->pad()->right(pad_right);

  // run interpreter
  locomotiv::NodeExecution::get().run(dw_conv2d);

  // get result of calculation
  auto dw_conv2d_result = locomotiv::annot_data(dw_conv2d);

  // check the result
  ASSERT_NE(dw_conv2d_result, nullptr);
  ASSERT_TRUE(dw_conv2d_result->dtype() == loco::DataType::FLOAT32);
  ASSERT_TRUE(*(dw_conv2d_result->shape()) == ofm_shape);

  auto ofm_overlay =
      make_overlay<float, LexicalLayout>(ofm_shape, const_cast<float *>(expected_ofm));
  for (nncc::core::ADT::tensor::IndexEnumerator e{ofm_shape}; e.valid(); e.advance())
  {
    const auto &ind = e.current();
    ASSERT_FLOAT_EQ(ofm_overlay.at(ind), dw_conv2d_result->as_f32_bufptr()->at(ind));
  }

  ASSERT_EQ(loco::Domain::Feature, locomotiv::annot_domain(dw_conv2d));
}

} // namespace

// clang-format off

/* ifm, ker and ofm are from the code below:

ifm = tf.random_normal([1, 5, 5, 2], stddev=1.1)
ker = tf.random_normal([4, 4, 2, 3], stddev=1.1)
out = tf.nn.depthwise_conv2d(ifm, ker, strides = [1, 1, 1, 1], padding= 'VALID')

with tf.Session() as sess:
    print(sess.run(out))
*/
TEST(NodeExecution_DepthwiseConv2D, f32_random_valid)
{
  using nncc::core::ADT::tensor::Shape;

  const float ifm[] = {0.8122538,   1.209147,    0.6903842,   -0.26646265, 1.516799,    -1.8540707,
                       -0.74240327, 1.7811562,   -0.03699546, -0.44468504, -1.4982721,  -1.1858582,
                       -0.21140318, -0.974522,   1.0000849,   -1.294535,   -0.6108882,  0.25827602,
                       1.3631831,   -0.5180266,  0.20870179,  0.18333802,  -0.42263857, -1.6694735,
                       0.0415236,   -0.3903758,  2.0933757,   -0.29660916, 2.1218338,   -1.1599928,
                       0.57163256,  0.48865932,  -1.3622656,  0.35924262,  1.2951899,   -0.1769997,
                       0.74513537,  -0.31920406, -1.2902768,  -0.7095059,  1.9157801,   -0.41028237,
                       1.2502829,   0.3354887,   1.4199319,   -0.20366786, -0.8828556,  0.5173567,
                       1.7708117,   -0.30096334};
  const float ker[] = {
      -0.19805557, 0.58464956,  -0.7804337,  0.06974592,  0.45790604,  0.24833807,  0.43393376,
      0.2541043,   -0.04406675, -0.32167575, 1.0546446,   -1.4978354,  0.20829494,  1.1659569,
      0.37908667,  -0.94137955, 0.293349,    -1.1023049,  0.76133233,  0.55595005,  1.4458209,
      1.6128604,   1.5655615,   -2.183877,   -0.90535915, -0.49858555, 1.7168728,   -1.1590382,
      0.6706056,   1.2215618,   -0.06603386, 0.16559464,  0.541991,    -0.44488335, 0.766181,
      1.0227629,   -0.6352362,  -1.670828,   -0.63334507, 0.0313305,   -0.6721083,  0.50112915,
      -0.15218066, 0.67222077,  -0.3613627,  -0.08516614, -0.5024078,  -0.9503976,  -2.1892295,
      1.8308185,   -0.15187284, 1.5761136,   0.24869336,  -1.7378871,  -0.22518761, 1.0175673,
      0.7084485,   -0.74157554, -1.8185995,  -1.3330095,  -0.04427439, 1.0556892,   -0.68243974,
      0.32001218,  2.0901792,   -1.1612813,  0.7294674,   0.05740008,  -0.00832882, 1.0446658,
      0.4477195,   -0.09174404, -1.0176039,  1.5066665,   -2.148343,   0.29421416,  0.93011874,
      -0.15737922, -1.6444012,  0.25780794,  -0.6545867,  -0.3488956,  0.26167992,  -0.154414,
      0.2798124,   -0.8590068,  2.0494444,   0.48268002,  0.81941164,  -0.4848027,  0.76870304,
      0.7102261,   0.45778143,  0.23214905,  -0.17742023, -0.75016516};
  const float ofm[] = {4.474646,   0.6792067, -1.9799856, 7.484751,   4.3087378,  -1.905938,
                       1.4887369,  0.4361322, 0.79539883, -3.8583446, -4.502204,  4.356392,
                       -5.3030324, 3.493003,  -4.349277,  2.3069482,  -3.8881323, -0.73901534,
                       -0.6629516, 2.1247253, -4.9229584, 1.6716996,  -3.0208125, 1.0597891};

  run_test(ifm, ker, ofm,
           Shape{1, 5, 5, 2}, Shape{4, 4, 2, 3}, Shape{1, 2, 2, 6}, // shapes of input, ker, output
           1, 1  // stride
           );
}

// TODO Add same padding test

// clang-format on
