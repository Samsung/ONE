/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/CommonSubExpressionEliminationPass.h"

#include "PassTestGraphs.h"

#include <luci/IR/CircleNodes.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci;
using namespace luci::test;

class QuantizeGraphlet
{
public:
  QuantizeGraphlet() = default;

  virtual ~QuantizeGraphlet() = default;

  void init(loco::Graph *g)
  {
    _quantize = g->nodes()->create<luci::CircleQuantize>();

    auto qparam = std::make_unique<luci::CircleQuantParam>();
    {
      qparam->scale.emplace_back(1.0);
      qparam->zerop.emplace_back(128);
    }
    _quantize->name("quantize");
    _quantize->quantparam(std::move(qparam));
    _quantize->dtype(loco::DataType::S16);
  }

protected:
  luci::CircleQuantize *_quantize = nullptr;
};

class CSE_QuantizeTestGraph : public CommonSubExpressionEliminationTestGraph,
                              public QuantizeGraphlet
{
public:
  std::vector<luci::CircleQuantize *> ops;

protected:
  virtual loco::Node *createExpression(luci::CircleNode *ifm, const std::string &name) override
  {
    auto expr = g()->nodes()->create<luci::CircleQuantize>();

    auto qparam = std::make_unique<luci::CircleQuantParam>();
    {
      qparam->scale.emplace_back(1.0);
      qparam->zerop.emplace_back(128);
    }
    expr->name(name + "_quantize");
    expr->quantparam(std::move(qparam));
    expr->dtype(loco::DataType::S16);
    expr->shape({1, 8, 8, 32});

    expr->input(ifm);

    ops.emplace_back(expr);

    // Set ifm dtype as uint8
    ifm->dtype(loco::DataType::U8);

    return expr;
  };

public:
  void init(void)
  {
    CommonSubExpressionEliminationTestGraph::init({{1, 8, 8, 32}}, {{1, 8, 8, 32}, {1, 8, 8, 32}});
  }
};

class CSE_TransposeTestGraph : public CommonSubExpressionEliminationTestGraph
{
public:
  std::vector<luci::CircleTranspose *> ops;

protected:
  virtual loco::Node *createExpression(luci::CircleNode *ifm, const std::string &name) override
  {
    auto perm = g()->nodes()->create<luci::CircleConst>();
    perm->name(name + "_perm");
    perm->dtype(loco::DataType::S32);
    perm->shape({4});
    perm->size<loco::DataType::S32>(4);
    perm->at<loco::DataType::S32>(0) = 0;
    perm->at<loco::DataType::S32>(1) = 3;
    perm->at<loco::DataType::S32>(2) = 1;
    perm->at<loco::DataType::S32>(3) = 2;

    auto expr = g()->nodes()->create<luci::CircleTranspose>();
    expr->name(name + "_transpose");
    expr->dtype(loco::DataType::FLOAT32);
    expr->shape({1, 32, 8, 8});
    expr->a(ifm);
    expr->perm(perm);

    ops.emplace_back(expr);

    return expr;
  };

public:
  void init(void)
  {
    CommonSubExpressionEliminationTestGraph::init({{1, 8, 8, 32}}, {{1, 32, 8, 8}, {1, 32, 8, 8}});
  }
};

} // namespace

TEST(CommonSubExpressionEliminationTest, Quantize)
{
  CSE_QuantizeTestGraph g;
  luci::CommonSubExpressionEliminationPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(CommonSubExpressionEliminationTest, Quantize_NEG)
{
  CSE_QuantizeTestGraph g;

  luci::CommonSubExpressionEliminationPass pass;

  g.init();

  // Different pattern
  g.ops[0]->input(g.ops[2]);

  EXPECT_FALSE(pass.run(g.g()));
}

TEST(CommonSubExpressionEliminationTest, Transpose)
{
  CSE_TransposeTestGraph g;
  luci::CommonSubExpressionEliminationPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(CommonSubExpressionEliminationTest, Transpose_NEG)
{
  CSE_TransposeTestGraph g;

  luci::CommonSubExpressionEliminationPass pass;

  g.init();

  // Different pattern
  g.ops[0]->a(g.ops[2]);

  EXPECT_FALSE(pass.run(g.g()));
}
