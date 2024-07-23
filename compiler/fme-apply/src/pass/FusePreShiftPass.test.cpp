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

#include "FusePreShiftPass.h"
#include "Support.Cast.h"

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

using namespace fme_apply;

namespace
{

luci::CircleConst *create_const_node(loco::Graph *g, const loco::DataType dtype,
                                     const std::vector<uint32_t> &shape,
                                     const std::vector<float> &values)
{
  auto node = g->nodes()->create<luci::CircleConst>();
  node->dtype(dtype);
  node->rank(shape.size());

  uint32_t size = 1;
  for (uint32_t i = 0; i < shape.size(); ++i)
  {
    node->dim(i) = shape[i];
    size *= shape[i];
  }
  node->shape_status(luci::ShapeStatus::VALID);

  assert(values.size() == size); // FIX_CALLER_UNLESS

  node->size<loco::DataType::FLOAT32>(size);
  for (uint32_t i = 0; i < values.size(); ++i)
    node->at<loco::DataType::FLOAT32>(i) = values[i];

  return node;
}

/**
 *  PreShift-Instnorm graphlet
 *
 *   [PreShift]
 *       |
 *   [Instnorm]
 *
 */
class PreShiftInstnormGraphlet
{
public:
  void init(loco::Graph *g)
  {
    _preshift = g->nodes()->create<luci::CircleCustom>(2 /* arity */, 1 /* out */);
    _preshift->dtype(loco::DataType::FLOAT32);
    _preshift->inputs(
      1, create_const_node(g, loco::DataType::FLOAT32, {3} /* shape */, {2, 2, 2} /* value */));
    _preshift->shape({1, 4, 4, 3});
    _preshift->custom_code("PreShift");
    _preshift->name("prescale");

    _instnorm = g->nodes()->create<luci::CircleInstanceNorm>();
    _instnorm->input(_preshift);
    _instnorm->fusedActivationFunction(luci::FusedActFunc::NONE);
    _instnorm->dtype(loco::DataType::FLOAT32);
    _instnorm->shape({1, 4, 4, 3});
    _instnorm->name("instnorm");
  }

public:
  luci::CircleCustom *_preshift = nullptr;
  luci::CircleInstanceNorm *_instnorm = nullptr;
};

class PreShiftInstnormGraph : public luci::test::TestIOGraph, public PreShiftInstnormGraphlet
{
public:
  void init(void)
  {
    luci::test::TestIOGraph::init({1, 4, 4, 3}, {1, 4, 4, 3});
    PreShiftInstnormGraphlet::init(g());

    _preshift->inputs(0, input());

    output()->from(_instnorm);
  }

  std::unique_ptr<loco::Graph> graph(void) { return std::move(_g); }
};

} // namespace

TEST(FusePreShiftPassTest, preshift_instnorm)
{
  PreShiftInstnormGraph g;
  g.init();

  FusePreShiftPass fpsp;
  EXPECT_TRUE(fpsp.run(g.g()));

  auto instnorm = dynamic_cast<luci::CircleInstanceNorm *>(g.output()->from());
  EXPECT_NE(nullptr, instnorm);

  auto pre_shift = to_pre_shift(instnorm->input());
  EXPECT_EQ(nullptr, pre_shift); // No pre_shift
}

TEST(FusePreShiftPassTest, preshift_instnorm_NEG)
{
  PreShiftInstnormGraph g;
  g.init();
  g._instnorm->input(g.input());

  FusePreShiftPass fpsp;
  EXPECT_FALSE(fpsp.run(g.g()));
}
