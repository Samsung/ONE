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

x = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                shape=[1, 3, 3, 2], dtype=tf.float32)
y = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                shape=[1, 3, 3, 2], dtype=tf.float32)
out = tf.math.subtract(x, y)

with tf.Session() as sess:
    print(sess.run(out))
*/
TEST(NodeExecution_EltwiseSub, f32)
{
  float x_val[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  float y_val[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  float out_val[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  // make EltwiseSub(Pull, Pull)
  auto g = loco::make_graph();
  Shape input_shape{1, 3, 3, 2}; // NHWC

  auto inp_lhs = g->nodes()->create<loco::Pull>();
  {
    inp_lhs->dtype(loco::DataType::FLOAT32);
    inp_lhs->shape({1, 3, 3, 2});
  }

  auto inp_rhs = g->nodes()->create<loco::Pull>();
  {
    inp_rhs->dtype(loco::DataType::FLOAT32);
    inp_rhs->shape({1, 3, 3, 2});
  }

  auto eltwise_sub = g->nodes()->create<loco::EltwiseSub>();
  {
    eltwise_sub->lhs(inp_lhs);
    eltwise_sub->rhs(inp_rhs);
  }

  // Make and assign data to two pull nodes
  auto inp_lhs_buf = make_buffer<float, LexicalLayout>(input_shape);
  {
    int n = 0;
    for (IndexEnumerator e{inp_lhs_buf.shape()}; e.valid(); e.advance())
    {
      inp_lhs_buf.at(e.current()) = x_val[n++];
    }
  }

  auto inp_rhs_buf = make_buffer<float, LexicalLayout>(input_shape);
  {
    int n = 0;
    for (IndexEnumerator e{inp_rhs_buf.shape()}; e.valid(); e.advance())
    {
      inp_rhs_buf.at(e.current()) = y_val[n++];
    }
  }

  auto inp_lhs_data = locomotiv::make_data(inp_lhs_buf);
  locomotiv::annot_data(inp_lhs, std::move(inp_lhs_data));
  locomotiv::annot_domain(inp_lhs, loco::Domain::Tensor);

  auto inp_rhs_data = locomotiv::make_data(inp_rhs_buf);
  locomotiv::annot_data(inp_rhs, std::move(inp_rhs_data));
  locomotiv::annot_domain(inp_rhs, loco::Domain::Tensor);

  // run the network
  locomotiv::NodeExecution::get().run(eltwise_sub);

  // get result
  auto eltwise_sub_data = locomotiv::annot_data(eltwise_sub);

  // comparing the result
  ASSERT_NE(eltwise_sub_data, nullptr);
  ASSERT_EQ(loco::DataType::FLOAT32, eltwise_sub_data->dtype());
  ASSERT_EQ(Shape({1, 3, 3, 2}), *(eltwise_sub_data->shape()));

  uint32_t n = 0;
  for (IndexEnumerator e{*(eltwise_sub_data->shape())}; e.valid(); e.advance())
  {
    ASSERT_FLOAT_EQ(out_val[n++], eltwise_sub_data->as_f32_bufptr()->at(e.current()));
  }

  ASSERT_EQ(loco::Domain::Tensor, locomotiv::annot_domain(eltwise_sub));
}
