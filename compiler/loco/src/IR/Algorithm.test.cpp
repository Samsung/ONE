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

#include "loco/IR/Algorithm.h"
#include "loco/IR/Graph.h"

#include <algorithm>

#include <gtest/gtest.h>

namespace
{

bool contains(const std::vector<loco::Node *> &vec, loco::Node *val)
{
  return std::any_of(vec.begin(), vec.end(), [val](loco::Node *node) { return node == val; });
}

bool contains(const std::set<loco::Node *> &s, loco::Node *val)
{
  return std::any_of(s.begin(), s.end(), [val](loco::Node *node) { return node == val; });
}

} // namespace

TEST(AlgorithmTest, postorder_traversal)
{
  auto g = loco::make_graph();

  auto pull_1 = g->nodes()->create<loco::Pull>();
  auto push = g->nodes()->create<loco::Push>();

  push->from(pull_1);

  // Create a dummy node unreachable from the above "push" node
  g->nodes()->create<loco::Pull>();

  auto seq = loco::postorder_traversal({push});

  ASSERT_EQ(seq.size(), 2);
  ASSERT_EQ(seq.at(0), pull_1);
  ASSERT_EQ(seq.at(1), push);
}

TEST(AlgorithmTest, postorder_traversal_visit_once)
{
  auto g = loco::make_graph();

  // Create a network of the following form:
  //
  //   Push1  Push2 <-- outputs
  //    \     /
  //     Pull  <-- input
  //
  auto pull = g->nodes()->create<loco::Pull>();
  auto push_1 = g->nodes()->create<loco::Push>();
  auto push_2 = g->nodes()->create<loco::Push>();

  push_1->from(pull);
  push_2->from(pull);

  auto seq = loco::postorder_traversal({push_1, push_2});

  ASSERT_EQ(seq.size(), 3);
  ASSERT_TRUE(contains(seq, pull));
  ASSERT_TRUE(contains(seq, push_1));
  ASSERT_TRUE(contains(seq, push_2));
}

TEST(AlgorithmTest, postorder_traversal_incomplte_graph)
{
  auto g = loco::make_graph();

  // Create a network of the following form:
  //
  //       TensorConcat
  //      /            \
  //  Pull              X
  //
  auto pull = g->nodes()->create<loco::Pull>();
  auto concat = g->nodes()->create<loco::TensorConcat>();

  concat->lhs(pull);

  auto seq = loco::postorder_traversal({concat});

  ASSERT_EQ(seq.size(), 2);
  ASSERT_EQ(seq.at(0), pull);
  ASSERT_EQ(seq.at(1), concat);
}

TEST(AlgorithmTest, active_nodes)
{
  auto g = loco::make_graph();

  auto pull = g->nodes()->create<loco::Pull>();
  auto push = g->nodes()->create<loco::Push>();

  push->from(pull);

  // NOTE This new Push node is unnecessary to compute "push"
  g->nodes()->create<loco::Push>();

  auto s = loco::active_nodes({push});

  ASSERT_EQ(s.size(), 2);
  ASSERT_TRUE(contains(s, pull));
  ASSERT_TRUE(contains(s, push));
}
