/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseMeanWithMeanPass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

/**
 *  Simple graph for test
 *
 *  BEFORE
 *                  |
 *          [CircleMean, axis<1>]
 *                  |
 *         [CircleMean, axis<1>]
 *                  |
 *
 *  AFTER
 *                  |
 *          [CircleMean, axis<1,2>]
 *                  |
 *
 */
class MeansGraphlet
{
public:
  MeansGraphlet() = default;

  void init(loco::Graph *g)
  {
    _mean1 = g->nodes()->create<luci::CircleMean>();
    _mean2 = g->nodes()->create<luci::CircleMean>();
    _indices1 = g->nodes()->create<luci::CircleConst>();
    _indices2 = g->nodes()->create<luci::CircleConst>();

    _mean1->name("mean1");
    _mean2->name("mean2");
    _indices1->name("indices1");
    _indices2->name("indices2");
  }

public:
  luci::CircleMean *mean1() { return _mean1; }
  luci::CircleMean *mean2() { return _mean2; }

protected:
  luci::CircleMean *_mean1 = nullptr;
  luci::CircleMean *_mean2 = nullptr;
  luci::CircleConst *_indices1 = nullptr;
  luci::CircleConst *_indices2 = nullptr;
};

class FuseActTestGraph : public TestIOGraph, public MeansGraphlet
{
public:
  FuseActTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 64, 20, 32}, {1, 20});
    MeansGraphlet::init(g());

    _indices1->rank(1);
    _indices1->dtype(loco::DataType::S32);
    _indices1->size<loco::DataType::S32>(1);
    _indices1->at<loco::DataType::S32>(0) = static_cast<int32_t>(1);
    _indices1->shape_status(luci::ShapeStatus::VALID);

    _indices2->rank(1);
    _indices2->dtype(loco::DataType::S32);
    _indices2->size<loco::DataType::S32>(1);
    _indices2->at<loco::DataType::S32>(0) = static_cast<int32_t>(2);
    _indices2->shape_status(luci::ShapeStatus::VALID);

    _mean1->input(input());
    _mean1->reduction_indices(_indices1);

    _mean2->input(_mean1);
    _mean2->reduction_indices(_indices2);

    output()->from(_mean2);
  }
};

} // namespace

TEST(FuseMeanWithMeanPassTest, name)
{
  luci::FuseMeanWithMeanPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(FuseMeanWithMeanPassTest, fuse_mean_with_mean)
{
  FuseActTestGraph g;
  luci::FuseMeanWithMeanPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(FuseMeanWithMeanPassTest, fus_mean_with_mean_NEG)
{
  FuseActTestGraph g;
  luci::FuseMeanWithMeanPass pass;

  g.init();

  // Add CircleRelu operation between CircleMeans operations
  auto relu = g.g()->nodes()->create<luci::CircleRelu>();
  relu->name("relu");
  relu->features(g.mean1());
  g.mean2()->input(relu);

  // Due to the CircleRelu operation, pass will not be applied
  EXPECT_FALSE(pass.run(g.g()));
}
