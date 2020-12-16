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

#include "loco/IR/Graph.h"

#include <gtest/gtest.h>

namespace
{

/// @brief Mockup class for loco::NamedEntity
struct NamedElement final : private loco::NamedEntity
{
  LOCO_NAMED_ENTITY_EXPOSE;
};

} // namespace

TEST(NamedTest, constructor)
{
  NamedElement elem;

  ASSERT_EQ("", elem.name());
}

TEST(NamedTest, setter_and_getter)
{
  NamedElement elem;

  elem.name("name");
  ASSERT_EQ("name", elem.name());
}

TEST(DataTypedMixinTest, constructor)
{
  loco::Mixin<loco::Trait::DataTyped> mixin;

  ASSERT_EQ(loco::DataType::Unknown, mixin.dtype());
}

TEST(DataTypedMixinTest, setter_and_getter)
{
  loco::Mixin<loco::Trait::DataTyped> mixin;

  mixin.dtype(loco::DataType::FLOAT32);
  ASSERT_EQ(loco::DataType::FLOAT32, mixin.dtype());
}

TEST(TensorShapedMixinTest, setter_and_getter)
{
  loco::Mixin<loco::Trait::TensorShaped> mixin;

  mixin.shape({1, 2, 3, 4});
  ASSERT_NE(mixin.shape(), nullptr);
  ASSERT_EQ(4, mixin.shape()->rank());
  ASSERT_EQ(1, mixin.shape()->dim(0));
  ASSERT_EQ(2, mixin.shape()->dim(1));
  ASSERT_EQ(3, mixin.shape()->dim(2));
  ASSERT_EQ(4, mixin.shape()->dim(3));
}

TEST(GraphTest, create_and_destroy_node)
{
  auto g = loco::make_graph();

  auto pull = g->nodes()->create<loco::Pull>();

  ASSERT_NO_THROW(g->nodes()->destroy(pull));
  ASSERT_THROW(g->nodes()->destroy(pull), std::invalid_argument);
}

TEST(GraphTest, create_input)
{
  auto g = loco::make_graph();

  auto input = g->inputs()->create();

  // TODO Add more checks
  ASSERT_EQ(nullptr, input->shape());
  ASSERT_EQ(0, input->index());
}

TEST(GraphTest, create_output)
{
  auto g = loco::make_graph();

  auto output = g->outputs()->create();

  // TODO Add more checks
  ASSERT_EQ(nullptr, output->shape());
  ASSERT_EQ(0, output->index());
}

namespace
{
// temp node with multple params for ctor. loco::CanonicalOpcode::ReLU is used for simplicity
class ParamCtorNode
  : public loco::CanonicalNodeDef<loco::CanonicalOpcode::ReLU, loco::FixedArity<0>::Mixin>
{
public:
  ParamCtorNode(int i, float f)
  {
    _i = i;
    _f = f;
  }

  int i() { return _i; }
  float f() { return _f; }

private:
  int _i;
  float _f;
};
} // namespace

TEST(GraphTest, consturctor_with_param_node)
{
  auto g = loco::make_graph();

  auto test_node = g->nodes()->create<ParamCtorNode>(22, 11.11);

  ASSERT_EQ(g.get(), test_node->graph());
  ASSERT_EQ(g.get(), const_cast<const ParamCtorNode *>(test_node)->graph());

  ASSERT_EQ(22, test_node->i());
  ASSERT_FLOAT_EQ(test_node->f(), 11.11);

  ASSERT_NO_THROW(g->nodes()->destroy(test_node));
  ASSERT_THROW(g->nodes()->destroy(test_node), std::invalid_argument);
}

TEST(GraphTest, getters_over_const_instance)
{
  auto g = loco::make_graph();

  auto pull = g->nodes()->create<loco::Pull>();
  auto push = g->nodes()->create<loco::Push>();

  loco::link(g->inputs()->create(), pull);
  loco::link(g->outputs()->create(), push);

  auto ptr = const_cast<const loco::Graph *>(g.get());

  EXPECT_EQ(ptr->nodes()->size(), 2);
  EXPECT_EQ(ptr->inputs()->size(), 1);
}

TEST(GraphTest, graph_node_enumeration)
{
  auto g = loco::make_graph();

  auto pull_1 = g->nodes()->create<loco::Pull>();
  auto push_1 = g->nodes()->create<loco::Push>();

  auto nodes = loco::all_nodes(g.get());

  // Returns true if "nodes" includes a given node
  auto member = [&nodes](loco::Node *node) { return nodes.find(node) != nodes.end(); };

  ASSERT_EQ(2, nodes.size());
  ASSERT_TRUE(member(pull_1));
  ASSERT_TRUE(member(push_1));
}

TEST(GraphTest, graph_inout_enumeration)
{
  auto g = loco::make_graph();

  std::vector<loco::Pull *> pull_nodes;

  auto pull_1 = g->nodes()->create<loco::Pull>();
  auto pull_2 = g->nodes()->create<loco::Pull>();
  auto pull_3 = g->nodes()->create<loco::Pull>();

  auto push_1 = g->nodes()->create<loco::Push>();
  auto push_2 = g->nodes()->create<loco::Push>();
  auto push_3 = g->nodes()->create<loco::Push>();

  loco::link(g->inputs()->create(), pull_2);
  loco::link(g->inputs()->create(), pull_1);

  loco::link(g->outputs()->create(), push_1);
  loco::link(g->outputs()->create(), push_3);

  auto output_nodes = loco::output_nodes(g.get());

  ASSERT_EQ(2, output_nodes.size());
  ASSERT_EQ(push_1, output_nodes.at(0));
  ASSERT_EQ(push_3, output_nodes.at(1));
}

TEST(GraphTest, graph_name)
{
  auto g = loco::make_graph();

  g->name("HelloGraph");
  ASSERT_TRUE(g->name() == "HelloGraph");
}

TEST(GraphTest, graph_name_nullptr_NEG)
{
  auto g = loco::make_graph();

  EXPECT_ANY_THROW(g->name(nullptr));
}
