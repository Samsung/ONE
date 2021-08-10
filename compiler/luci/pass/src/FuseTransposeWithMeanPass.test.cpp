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

#include "luci/Pass/FuseTransposeWithMeanPass.h"

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
 *          [CircleTranspose, perm<0, 2, 3, 1>]
 *                  |
 *          [CircleMean, axis<3>]
 *                  |
 *
 *  AFTER
 *                  |
 *          [CircleMean, axis<1>]       [CircleTranspose, perm<0, 2, 3, 1>]
 *                  |                            |
 *                                      [CircleMean, axis<3>]
 *
 */
class FuseTransposeWithMeanTestGraph : public TestIOGraph
{
public:
  FuseTransposeWithMeanTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 64, 20, 32}, {1, 20, 32});

    _mean = g()->nodes()->create<luci::CircleMean>();
    _transpose = g()->nodes()->create<luci::CircleTranspose>();
    _indices = g()->nodes()->create<luci::CircleConst>();
    _perm = g()->nodes()->create<luci::CircleConst>();

    _mean->name("mean");
    _transpose->name("transpose");
    _indices->name("indices");
    _perm->name("perm");

    _indices->rank(1);
    _indices->dtype(loco::DataType::S32);
    _indices->size<loco::DataType::S32>(1);
    _indices->at<loco::DataType::S32>(0) = static_cast<int32_t>(3);
    _indices->dim(0) = 1;
    _indices->shape_status(luci::ShapeStatus::VALID);

    _perm->rank(1);
    _perm->dtype(loco::DataType::S32);
    _perm->size<loco::DataType::S32>(4);
    _perm->dim(0) = 4;
    _perm->at<loco::DataType::S32>(0) = static_cast<int32_t>(0);
    _perm->at<loco::DataType::S32>(1) = static_cast<int32_t>(2);
    _perm->at<loco::DataType::S32>(2) = static_cast<int32_t>(3);
    _perm->at<loco::DataType::S32>(3) = static_cast<int32_t>(1);
    _perm->shape_status(luci::ShapeStatus::VALID);

    _transpose->a(input());
    _transpose->perm(_perm);

    _mean->input(_transpose);
    _mean->reduction_indices(_indices);


    output()->from(_mean);
  }

  luci::CircleTranspose *transpose(void) const { return _transpose; }
  luci::CircleMean *mean(void) const { return _mean; }

private:
  luci::CircleTranspose *_transpose = nullptr;
  luci::CircleMean *_mean = nullptr;
  luci::CircleConst *_indices = nullptr;
  luci::CircleConst *_perm = nullptr;
};

} // namespace

TEST(FuseTransposeWithMeanPassTest, name)
{
  luci::FuseTransposeWithMeanPass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

TEST(FuseTransposeWithMeanPassTest, fuse_mean_with_mean)
{
  FuseTransposeWithMeanTestGraph g;
  luci::FuseTransposeWithMeanPass pass;

  g.init();

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(FuseTransposeWithMeanPassTest, fus_mean_with_mean_NEG)
{
  FuseTransposeWithMeanTestGraph g;
  luci::FuseTransposeWithMeanPass pass;

  g.init();

  // Add CircleRelu operation between CircleMeans operations
  auto relu = g.g()->nodes()->create<luci::CircleRelu>();
  relu->name("relu");
  relu->features(g.transpose());
  g.mean()->input(relu);

  // Due to the CircleRelu operation, pass will not be applied
  EXPECT_FALSE(pass.run(g.g()));
}
