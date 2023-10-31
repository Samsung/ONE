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

#include "PatternResolver.h"

#include "core/TestHelper.h"

#include <luci/CircleQuantizer.h>
#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <cmath>
#include <gtest/gtest.h>

using LayerParam = luci::CircleQuantizer::Options::LayerParam;

namespace
{

using namespace luci::test;

class LayerNormGraphlet
{
public:
  LayerNormGraphlet() = default;
  virtual ~LayerNormGraphlet() = default;

  void init(loco::Graph *g)
  {
    ifm = nullptr;

    ifm = g->nodes()->create<luci::CircleAbs>();
    mean_of_ifm = g->nodes()->create<luci::CircleMean>();
    sub = g->nodes()->create<luci::CircleSub>();
    sub_squared = g->nodes()->create<luci::CircleMul>();
    mean_as_variance = g->nodes()->create<luci::CircleMean>();
    add_eps = g->nodes()->create<luci::CircleAdd>();
    rsqrt = g->nodes()->create<luci::CircleRsqrt>();
    mul = g->nodes()->create<luci::CircleMul>();
    _eps = g->nodes()->create<luci::CircleConst>();
    _mean_of_ifm_indices = g->nodes()->create<luci::CircleConst>();
    _mean_as_variance_indices = g->nodes()->create<luci::CircleConst>();

    ifm->name("ifm");
    mean_of_ifm->name("mean_of_ifm");
    sub->name("sub");
    sub_squared->name("sub_squared");
    mean_as_variance->name("mean_as_variance");
    add_eps->name("add_eps");
    rsqrt->name("rsqrt");
    mul->name("mul");
    _eps->name("eps");
    _mean_of_ifm_indices->name("mean_of_ifm_indices");
    _mean_as_variance_indices->name("mean_as_variance_indices");

    _eps->dtype(loco::DataType::FLOAT32);
    _eps->size<loco::DataType::FLOAT32>(1);
    _eps->shape({1});
    _eps->at<loco::DataType::FLOAT32>(0) = 1.e-05f;
    _eps->shape_status(luci::ShapeStatus::VALID);

    _mean_of_ifm_indices->dtype(loco::DataType::S32);
    _mean_of_ifm_indices->size<loco::DataType::S32>(1);
    _mean_of_ifm_indices->shape({1});
    _mean_of_ifm_indices->at<loco::DataType::S32>(0) = -1;
    _mean_of_ifm_indices->shape_status(luci::ShapeStatus::VALID);

    _mean_as_variance_indices->dtype(loco::DataType::S32);
    _mean_as_variance_indices->size<loco::DataType::S32>(1);
    _mean_as_variance_indices->shape({1});
    _mean_as_variance_indices->at<loco::DataType::S32>(0) = -1;
    _mean_as_variance_indices->shape_status(luci::ShapeStatus::VALID);
  }

public:
  luci::CircleAbs *ifm = nullptr;
  luci::CircleMean *mean_of_ifm = nullptr;
  luci::CircleSub *sub = nullptr;
  luci::CircleMul *sub_squared = nullptr;
  luci::CircleMean *mean_as_variance = nullptr;
  luci::CircleAdd *add_eps = nullptr;
  luci::CircleRsqrt *rsqrt = nullptr;
  luci::CircleMul *mul = nullptr;

protected:
  luci::CircleConst *_eps = nullptr;
  luci::CircleConst *_mean_of_ifm_indices = nullptr;
  luci::CircleConst *_mean_as_variance_indices = nullptr;
};

class LayerNormTestGraph : public TestIOGraph, public LayerNormGraphlet
{
public:
  LayerNormTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 12, 11, 15}, {1, 12, 11, 15});
    LayerNormGraphlet::init(g());

    ifm->x(input());
    mean_of_ifm->input(ifm);
    mean_of_ifm->reduction_indices(_mean_of_ifm_indices);
    sub->x(ifm);
    sub->y(mean_of_ifm);
    sub_squared->x(sub);
    sub_squared->y(sub);
    mean_as_variance->input(sub_squared);
    mean_as_variance->reduction_indices(_mean_as_variance_indices);
    add_eps->x(mean_as_variance);
    add_eps->y(_eps);
    rsqrt->x(add_eps);
    mul->x(sub);
    mul->y(rsqrt);

    output()->from(mul);
  }
};

} // namespace

TEST(LayerNormPatternResolverTest, resolve_pattern)
{
  auto m = luci::make_module();
  LayerNormTestGraph g;
  g.init();
  g.transfer_to(m.get());

  std::map<luci::CircleNode *, LayerParam> params;
  mpqsolver::pattern::Q8LayerNormWithQ16VarianceResolver resolver;
  EXPECT_NO_THROW({ params = resolver.resolve(m.get()); });

  std::set<luci::CircleNode *> q16_nodes = {g.sub_squared, g.mean_as_variance, g.add_eps, g.rsqrt};
  std::set<luci::CircleNode *> q8_nodes = {g.mean_of_ifm, g.sub, g.mul};

  // params of all valid layers are set
  EXPECT_EQ(params.size(), q16_nodes.size() + q8_nodes.size());

  for (auto param : params)
  {
    // params of all layers are set as prescribed
    if (q16_nodes.find(param.first) != q16_nodes.end())
    {
      EXPECT_STREQ(param.second.dtype.c_str(), "int16");
    }
    else if (q8_nodes.find(param.first) != q8_nodes.end())
    {
      EXPECT_STREQ(param.second.dtype.c_str(), "uint8");
    }
  }
}

TEST(LayerNormPatternResolverTest, resolve_pattern_NEG)
{
  std::map<luci::CircleNode *, LayerParam> params;
  mpqsolver::pattern::Q8LayerNormWithQ16VarianceResolver resolver;
  EXPECT_ANY_THROW(resolver.resolve(nullptr));
}

TEST(SoftmaxResolverTest, resolve_pattern)
{
  auto m = luci::make_module();
  mpqsolver::test::models::SoftmaxTestGraph g;
  g.init();
  g.transfer_to(m.get());

  std::map<luci::CircleNode *, LayerParam> params;
  mpqsolver::pattern::Q8SoftmaxWithQ16SubExpResolver resolver;
  EXPECT_NO_THROW({ params = resolver.resolve(m.get()); });

  std::set<luci::CircleNode *> q16_nodes = {g._sub, g._exp};
  std::set<luci::CircleNode *> q8_nodes = {g._ifm, g._max, g._sum, g._div};

  // params of all valid layers are set
  EXPECT_EQ(params.size(), q16_nodes.size() + q8_nodes.size());

  for (auto param : params)
  {
    // params of all layers are set as prescribed
    if (q16_nodes.find(param.first) != q16_nodes.end())
    {
      EXPECT_STREQ(param.second.dtype.c_str(), "int16");
    }
    else if (q8_nodes.find(param.first) != q8_nodes.end())
    {
      EXPECT_STREQ(param.second.dtype.c_str(), "uint8");
    }
  }
}

TEST(SoftmaxPatternResolverTest, resolve_pattern_NEG)
{
  mpqsolver::pattern::Q8SoftmaxWithQ16SubExpResolver resolver;
  EXPECT_ANY_THROW(resolver.resolve(nullptr));
}
