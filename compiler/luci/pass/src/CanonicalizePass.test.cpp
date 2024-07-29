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

#include "luci/Pass/CanonicalizePass.h"

#include <luci/IR/CircleNodes.h>

#include <luci/test/TestIOGraph.h>

#include <vector>

#include <gtest/gtest.h>

TEST(CanonicalizePassTest, name)
{
  luci::CanonicalizePass pass;
  auto const name = pass.name();
  ASSERT_NE(nullptr, name);
}

namespace
{

using namespace luci::test;

struct PadGraphlet
{
  PadGraphlet() = default;

  void init(loco::Graph *g)
  {
    _pad = g->nodes()->create<luci::CirclePad>();
    _padv2 = g->nodes()->create<luci::CirclePadV2>();
    _paddings_s32 = g->nodes()->create<luci::CircleConst>();
    _paddings_s64 = g->nodes()->create<luci::CircleConst>();
    // NOTE PadV2.constant_values is not set as test doesn't use this

    _pad->name("pad");
    _padv2->name("padv2");
    _paddings_s32->name("paddings_s32");
    _paddings_s64->name("paddings_s64");

    _paddings_s64->dtype(loco::DataType::S64);
    _paddings_s64->rank(2);
    _paddings_s64->dim(0).set(4);
    _paddings_s64->dim(1).set(2);
    _paddings_s64->shape_status(luci::ShapeStatus::VALID);

    _paddings_s32->dtype(loco::DataType::S32);
    _paddings_s32->rank(2);
    _paddings_s32->dim(0).set(4);
    _paddings_s32->dim(1).set(2);
    _paddings_s32->shape_status(luci::ShapeStatus::VALID);

    std::vector<int64_t> ps = {0, 0, 1, 1, 1, 1, 0, 0};

    uint32_t num_elements = static_cast<uint32_t>(ps.size());
    _paddings_s64->size<loco::DataType::S64>(num_elements);
    for (uint32_t i = 0; i < num_elements; i++)
      _paddings_s64->at<loco::DataType::S64>(i) = ps[i];

    _paddings_s32->size<loco::DataType::S32>(num_elements);
    for (uint32_t i = 0; i < num_elements; i++)
      _paddings_s32->at<loco::DataType::S32>(i) = static_cast<int32_t>(ps[i]);
  }

  luci::CirclePad *_pad = nullptr;
  luci::CirclePadV2 *_padv2 = nullptr;
  luci::CircleConst *_paddings_s32 = nullptr;
  luci::CircleConst *_paddings_s64 = nullptr;
};

class CanonicalizePadTestGraph : public TestIOGraph, public PadGraphlet
{
public:
  CanonicalizePadTestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 3, 3, 2}, {1, 5, 5, 2});
    PadGraphlet::init(g());

    _pad->input(input());
    _pad->paddings(_paddings_s64);

    output()->from(_pad);
  }
};

class CanonicalizePadV2TestGraph : public TestIOGraph, public PadGraphlet
{
public:
  CanonicalizePadV2TestGraph() = default;

  void init(void)
  {
    TestIOGraph::init({1, 3, 3, 2}, {1, 5, 5, 2});
    PadGraphlet::init(g());

    _padv2->input(input());
    _padv2->paddings(_paddings_s64);

    output()->from(_padv2);
  }
};

} // namespace

TEST(CanonicalizePassPadTest, paddings_64_to_32)
{
  CanonicalizePadTestGraph g;
  luci::CanonicalizePass pass;

  g.init();

  luci::CircleConst *paddings = dynamic_cast<luci::CircleConst *>(g._pad->paddings());
  EXPECT_NE(nullptr, paddings);
  EXPECT_EQ(paddings->dtype(), loco::DataType::S64);

  EXPECT_TRUE(pass.run(g.g()));

  paddings = dynamic_cast<luci::CircleConst *>(g._pad->paddings());
  EXPECT_NE(nullptr, paddings);
  EXPECT_EQ(paddings->dtype(), loco::DataType::S32);
}

TEST(CanonicalizePassPadTest, paddings_32_NEG)
{
  CanonicalizePadTestGraph g;
  luci::CanonicalizePass pass;

  g.init();
  g._pad->paddings(g._paddings_s32);

  luci::CircleConst *paddings = dynamic_cast<luci::CircleConst *>(g._pad->paddings());
  EXPECT_NE(nullptr, paddings);
  EXPECT_EQ(paddings->dtype(), loco::DataType::S32);

  EXPECT_FALSE(pass.run(g.g()));

  paddings = dynamic_cast<luci::CircleConst *>(g._pad->paddings());
  EXPECT_NE(nullptr, paddings);
  EXPECT_EQ(paddings->dtype(), loco::DataType::S32);
}

TEST(CanonicalizePassPadTest, paddings_32_over_NEG)
{
  CanonicalizePadTestGraph g;
  luci::CanonicalizePass pass;

  g.init();
  g._paddings_s64->at<loco::DataType::S64>(2) =
    static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 100;

  EXPECT_FALSE(pass.run(g.g()));

  luci::CircleConst *paddings = dynamic_cast<luci::CircleConst *>(g._pad->paddings());
  EXPECT_NE(nullptr, paddings);
  EXPECT_EQ(paddings->dtype(), loco::DataType::S64);
}

TEST(CanonicalizePassPadV2Test, paddings_64_to_32)
{
  CanonicalizePadV2TestGraph g;
  luci::CanonicalizePass pass;

  g.init();

  luci::CircleConst *paddings = dynamic_cast<luci::CircleConst *>(g._padv2->paddings());
  EXPECT_NE(nullptr, paddings);
  EXPECT_EQ(paddings->dtype(), loco::DataType::S64);

  EXPECT_TRUE(pass.run(g.g()));

  paddings = dynamic_cast<luci::CircleConst *>(g._padv2->paddings());
  EXPECT_NE(nullptr, paddings);
  EXPECT_EQ(paddings->dtype(), loco::DataType::S32);
}

TEST(CanonicalizePassPadV2Test, paddings_32_NEG)
{
  CanonicalizePadV2TestGraph g;
  luci::CanonicalizePass pass;

  g.init();
  g._padv2->paddings(g._paddings_s32);

  luci::CircleConst *paddings = dynamic_cast<luci::CircleConst *>(g._padv2->paddings());
  EXPECT_NE(nullptr, paddings);
  EXPECT_EQ(paddings->dtype(), loco::DataType::S32);

  EXPECT_FALSE(pass.run(g.g()));

  paddings = dynamic_cast<luci::CircleConst *>(g._padv2->paddings());
  EXPECT_NE(nullptr, paddings);
  EXPECT_EQ(paddings->dtype(), loco::DataType::S32);
}

TEST(CanonicalizePassPadV2Test, paddings_32_over_NEG)
{
  CanonicalizePadV2TestGraph g;
  luci::CanonicalizePass pass;

  g.init();
  g._paddings_s64->at<loco::DataType::S64>(2) =
    static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 100;

  EXPECT_FALSE(pass.run(g.g()));

  luci::CircleConst *paddings = dynamic_cast<luci::CircleConst *>(g._padv2->paddings());
  EXPECT_NE(nullptr, paddings);
  EXPECT_EQ(paddings->dtype(), loco::DataType::S64);
}
