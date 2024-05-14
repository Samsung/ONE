/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseRsqrtPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <cmath>
#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

std::unique_ptr<luci::CircleQuantParam> gen_qparam(const std::vector<float> &s,
                                                   const std::vector<int64_t> &zp)
{
  auto qparam = std::make_unique<luci::CircleQuantParam>();
  {
    for (auto scale : s)
      qparam->scale.push_back(scale);

    for (auto zerop : zp)
      qparam->zerop.push_back(zerop);
  }

  return std::move(qparam);
}

class S16RsqrtGraphlet
{
public:
  S16RsqrtGraphlet() = default;

  void init(loco::Graph *g)
  {
    _sqrt = g->nodes()->create<luci::CircleSqrt>();
    _div = g->nodes()->create<luci::CircleDiv>();
    _div_const = g->nodes()->create<luci::CircleConst>();

    _div->fusedActivationFunction(luci::FusedActFunc::NONE);

    _sqrt->dtype(loco::DataType::S16);
    _div->dtype(loco::DataType::S16);
    _div_const->dtype(loco::DataType::S16);

    _div_const->size<loco::DataType::S16>(1);
    _div_const->shape({1});
    _div_const->at<loco::DataType::S16>(0) = 1;
    _div_const->shape_status(luci::ShapeStatus::VALID);

    _sqrt->quantparam(gen_qparam({1.0}, {0}));
    _div->quantparam(gen_qparam({2.0}, {0}));
    _div_const->quantparam(gen_qparam({1.0}, {0}));
  }

  void invalid_act() { _div->fusedActivationFunction(luci::FusedActFunc::RELU); }

protected:
  luci::CircleSqrt *_sqrt = nullptr;
  luci::CircleDiv *_div = nullptr;
  luci::CircleConst *_div_const = nullptr;
};

class FuseS16RsqrtTestGraph : public TestIOGraph, public S16RsqrtGraphlet
{
public:
  FuseS16RsqrtTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1}, {1});
    S16RsqrtGraphlet::init(g());

    _sqrt->x(input());
    _div->x(_div_const);
    _div->y(_sqrt);

    output()->from(_div);
  }
};

} // namespace

TEST(FuseRsqrtPassTest, s16)
{
  FuseS16RsqrtTestGraph g;
  luci::FuseRsqrtPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(FuseRsqrtPassTest, fuse_invalid_act_NEG)
{
  FuseS16RsqrtTestGraph g;
  luci::FuseRsqrtPass pass;

  g.init();
  g.invalid_act();

  EXPECT_FALSE(pass.run(g.g()));
}
