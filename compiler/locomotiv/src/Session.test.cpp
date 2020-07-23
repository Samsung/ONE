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

#include "locomotiv/Session.h"
#include "locomotiv/NodeData.h"

#include "UserData.h"

#include <loco.h>
#include <nncc/core/ADT/tensor/Shape.h>
#include <nncc/core/ADT/tensor/Buffer.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <array>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::make_buffer;
using nncc::core::ADT::tensor::Shape;

TEST(Session, graph_IO_size)
{
  // Make graph
  auto g = loco::make_graph();

  // inputs
  const uint32_t inputs = 2;
  for (uint32_t i = 0; i < inputs; ++i)
  {
    auto pull = g->nodes()->create<loco::Pull>();
    loco::link(g->inputs()->create(), pull);
  }

  // outputs
  const uint32_t outputs = 3;
  for (uint32_t o = 0; o < outputs; ++o)
  {
    auto push = g->nodes()->create<loco::Push>();
    loco::link(g->outputs()->create(), push);
  }

  // Make session
  locomotiv::Session s(g.get());

  ASSERT_EQ(inputs, s.input_size());
  ASSERT_EQ(outputs, s.output_size());
}

TEST(Session, set_input)
{
  // Make graph
  auto g = loco::make_graph();
  auto pull = g->nodes()->create<loco::Pull>();
  pull->dtype(loco::DataType::FLOAT32);
  pull->rank(1);
  pull->dim(0) = 1;
  loco::link(g->inputs()->create(), pull);

  // Make good data
  auto buf = make_buffer<float, LexicalLayout>(Shape{1});
  auto data = locomotiv::make_data(buf);

  // Make data with different data type
  auto buf_not_dtype = make_buffer<int32_t, LexicalLayout>(Shape{1});
  auto data_not_dtype = locomotiv::make_data(buf_not_dtype);

  // Make data with different rank
  auto buf_not_rank = make_buffer<float, LexicalLayout>(Shape{1, 1});
  auto data_not_rank = locomotiv::make_data(buf_not_rank);

  // Make data with different dimension
  auto buf_not_dim = make_buffer<float, LexicalLayout>(Shape{2});
  auto data_not_dim = locomotiv::make_data(buf_not_dim);

  // Make session
  locomotiv::Session s(g.get());

  ASSERT_ANY_THROW(s.set_input(0, std::move(data_not_dtype)));
  ASSERT_ANY_THROW(s.set_input(0, std::move(data_not_rank)));
  ASSERT_ANY_THROW(s.set_input(0, std::move(data_not_dim)));
  ASSERT_NO_THROW(s.set_input(0, std::move(data)));
  ASSERT_ANY_THROW(s.set_input(0, std::move(data)));
}

TEST(Session, inference_identity)
{
  std::vector<std::unique_ptr<loco::Graph>> graphs;

  // pull-push / f32 / known shape
  {
    auto g = loco::make_graph();

    // Pull node
    auto pull_node = g->nodes()->create<loco::Pull>();
    pull_node->dtype(loco::DataType::FLOAT32);
    pull_node->rank(1);
    pull_node->dim(0) = 1;

    // Push node
    auto push_node = g->nodes()->create<loco::Push>();
    push_node->from(pull_node);

    // Input
    auto graph_input = g->inputs()->create();
    loco::link(graph_input, pull_node);

    // Output
    auto graph_output = g->outputs()->create();
    loco::link(graph_output, push_node);

    graphs.push_back(std::move(g));
  }

  // pull-push / f32 / unknown shape
  {
    auto g = loco::make_graph();

    // Pull node
    auto pull_node = g->nodes()->create<loco::Pull>();
    pull_node->dtype(loco::DataType::FLOAT32);
    pull_node->rank(1);
    pull_node->dim(0) = loco::make_dimension();

    // Push node
    auto push_node = g->nodes()->create<loco::Push>();
    push_node->from(pull_node);

    // Input
    auto graph_input = g->inputs()->create();
    loco::link(graph_input, pull_node);

    // Output
    auto graph_output = g->outputs()->create();
    loco::link(graph_output, push_node);

    graphs.push_back(std::move(g));
  }

  for (auto it = graphs.begin(); it != graphs.end(); ++it)
  {
    auto g = it->get();
    locomotiv::Session s(g);

    const Shape shape{1};
    auto buf = make_buffer<float, LexicalLayout>(shape);
    buf.at(Index{0}) = 3.14f;
    auto data = locomotiv::make_data(buf);

    // Input not ready
    ASSERT_ANY_THROW(s.infer());

    s.set_input(0, std::move(data));

    // Valid run
    ASSERT_NO_THROW(s.infer());
    // Multiple run is possible
    ASSERT_NO_THROW(s.infer());

    auto output_data = s.get_output(0);
    ASSERT_NE(output_data, nullptr);
    ASSERT_EQ(loco::DataType::FLOAT32, output_data->dtype());
    ASSERT_EQ(Shape{1}, *(output_data->shape()));
    ASSERT_EQ(3.14f, output_data->as_f32_bufptr()->at(Index{0}));
  }
}

TEST(Session, session_for_subgraph)
{
  /*
   * Make following graph:
   *   ConstGen_1 --
   *                \
   *   ConstGen_2 --- TensorConcat_1 --- TensorConcat_3 --- Push
   *                                   /
   *   ConstGen_3 --- TensorConcat_2 --
   *                /
   *   ConstGen_4 --
   */
  auto g = loco::make_graph();

  auto c1 = g->nodes()->create<loco::ConstGen>();
  auto c2 = g->nodes()->create<loco::ConstGen>();
  auto c3 = g->nodes()->create<loco::ConstGen>();
  auto c4 = g->nodes()->create<loco::ConstGen>();

  c1->dtype(loco::DataType::FLOAT32);
  c2->dtype(loco::DataType::FLOAT32);
  c3->dtype(loco::DataType::FLOAT32);
  c4->dtype(loco::DataType::FLOAT32);
  c1->shape({1});
  c2->shape({1});
  c3->shape({1});
  c4->shape({1});
  c1->size<loco::DataType::FLOAT32>(1);
  c2->size<loco::DataType::FLOAT32>(1);
  c3->size<loco::DataType::FLOAT32>(1);
  c4->size<loco::DataType::FLOAT32>(1);

  c1->at<loco::DataType::FLOAT32>(0) = 0.1f;
  c2->at<loco::DataType::FLOAT32>(0) = 0.2f;
  c3->at<loco::DataType::FLOAT32>(0) = 0.3f;
  c4->at<loco::DataType::FLOAT32>(0) = 0.4f;

  auto t1 = g->nodes()->create<loco::TensorConcat>();
  auto t2 = g->nodes()->create<loco::TensorConcat>();
  auto t3 = g->nodes()->create<loco::TensorConcat>();

  // Note: default concat axis is 0
  t1->lhs(c1);
  t1->rhs(c2);
  t2->lhs(c3);
  t2->rhs(c4);
  t3->lhs(t1);
  t3->rhs(t2);

  auto push = g->nodes()->create<loco::Push>();
  push->from(t3);

  {
    // Session to get t1 only
    locomotiv::Session s(g.get(), {t1});
    ASSERT_EQ(1, s.output_size());
    ASSERT_EQ(dynamic_cast<loco::Node *>(t1), s.get_output_node(0));

    s.infer();

    auto t1_data = s.get_output(0);
    ASSERT_NE(t1_data, nullptr);
    ASSERT_EQ(Shape{2}, *(t1_data->shape()));

    auto t1_buf = t1_data->as_f32_bufptr();
    ASSERT_EQ(0.1f, t1_buf->at({0}));
    ASSERT_EQ(0.2f, t1_buf->at({1}));
  }

  {
    // Session to get t2 only
    locomotiv::Session s(g.get(), {t2});
    ASSERT_EQ(1, s.output_size());
    ASSERT_EQ(dynamic_cast<loco::Node *>(t2), s.get_output_node(0));

    s.infer();

    auto t2_data = s.get_output(0);
    ASSERT_NE(t2_data, nullptr);
    ASSERT_EQ(Shape{2}, *(t2_data->shape()));

    auto t2_buf = t2_data->as_f32_bufptr();
    ASSERT_EQ(0.3f, t2_buf->at({0}));
    ASSERT_EQ(0.4f, t2_buf->at({1}));
  }

  {
    // Session to get t2 and push
    locomotiv::Session s(g.get(), {t2, push});
    ASSERT_EQ(2, s.output_size());
    ASSERT_EQ(dynamic_cast<loco::Node *>(t2), s.get_output_node(0));
    ASSERT_EQ(dynamic_cast<loco::Node *>(push), s.get_output_node(1));

    s.infer();

    auto t2_data = s.get_output(0);
    ASSERT_NE(t2_data, nullptr);
    ASSERT_EQ(Shape{2}, *(t2_data->shape()));

    auto t2_buf = t2_data->as_f32_bufptr();
    ASSERT_EQ(0.3f, t2_buf->at({0}));
    ASSERT_EQ(0.4f, t2_buf->at({1}));

    auto push_data = s.get_output(1);
    ASSERT_NE(push_data, nullptr);
    ASSERT_EQ(Shape{4}, *(push_data->shape()));

    auto push_buf = push_data->as_f32_bufptr();
    ASSERT_EQ(0.1f, push_buf->at({0}));
    ASSERT_EQ(0.2f, push_buf->at({1}));
    ASSERT_EQ(0.3f, push_buf->at({2}));
    ASSERT_EQ(0.4f, push_buf->at({3}));
  }
}

TEST(Session, ctor_by_range)
{
  // Make graph
  auto g = loco::make_graph();

  auto constgen = g->nodes()->create<loco::ConstGen>();
  auto relu = g->nodes()->create<loco::ReLU>();
  auto push = g->nodes()->create<loco::Push>();

  constgen->dtype(loco::DataType::FLOAT32);
  constgen->shape({2});
  constgen->size<loco::DataType::FLOAT32>(2);
  constgen->at<loco::DataType::FLOAT32>(0) = 0.1f;
  constgen->at<loco::DataType::FLOAT32>(1) = -0.1f;

  relu->input(constgen);
  push->from(relu);

  std::array<loco::Node *, 2> custom_outputs = {constgen, push};

  // Make Session by range
  locomotiv::Session s(g.get(), custom_outputs.begin(), custom_outputs.end());

  s.infer();

  auto constgen_data = s.get_output(0);
  ASSERT_NE(constgen_data, nullptr);
  ASSERT_EQ(Shape{2}, *(constgen_data->shape()));

  auto constgen_buf = constgen_data->as_f32_bufptr();
  ASSERT_EQ(0.1f, constgen_buf->at({0}));
  ASSERT_EQ(-0.1f, constgen_buf->at({1}));

  auto push_data = s.get_output(1);
  ASSERT_NE(push_data, nullptr);
  ASSERT_EQ(Shape{2}, *(push_data->shape()));

  auto push_buf = push_data->as_f32_bufptr();
  ASSERT_EQ(0.1f, push_buf->at({0}));
  ASSERT_EQ(0.0f, push_buf->at({1}));
}

// Below here is internal test for locomotiv, i.e. not public usage of locomotiv
#include "NodeDataImpl.h"
#include "NodeDomain.h"

TEST(Session, dtor)
{
  auto g = loco::make_graph();

  // Pull node
  auto pull = g->nodes()->create<loco::Pull>();
  pull->dtype(loco::DataType::FLOAT32);
  pull->rank(1);
  pull->dim(0) = 1;

  // Input
  auto input = g->inputs()->create();
  loco::link(input, pull);

  {
    locomotiv::Session s(g.get());

    auto buf = make_buffer<float, LexicalLayout>(Shape{1});
    auto data = locomotiv::make_data(buf);

    s.set_input(0, std::move(data));

    auto data_annotated = locomotiv::annot_data(pull);
    ASSERT_EQ(nullptr, data_annotated);
    auto user_data_annotated = locomotiv::user_data(pull);
    ASSERT_NE(user_data_annotated, nullptr);
    auto domain_annotated = locomotiv::annot_domain(pull);
    ASSERT_EQ(loco::Domain::Unknown, domain_annotated);
  }

  auto data_annotated = locomotiv::annot_data(pull);
  ASSERT_EQ(nullptr, data_annotated);
  auto user_data_annotated = locomotiv::user_data(pull);
  ASSERT_EQ(nullptr, user_data_annotated);
  auto domain_annotated = locomotiv::annot_domain(pull);
  ASSERT_EQ(loco::Domain::Unknown, domain_annotated);
}
