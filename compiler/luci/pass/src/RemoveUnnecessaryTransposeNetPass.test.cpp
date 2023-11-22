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

#include "luci/Pass/RemoveUnnecessaryTransposeNetPass.h"

#include <luci/IR/CircleNode.h>

#include <luci/test/TestIOGraph.h>

#include <gtest/gtest.h>

namespace
{

using namespace luci::test;

class TransposeReshapeTransposeGraph : public TestIOGraph
{

public:
  // create input-transpose-reshape-transpose-output graph
  void init_whole_graph(ShapeU32 in_shape, ShapeU32 front_perm, ShapeU32 mid_shape,
                        ShapeU32 back_perm, ShapeU32 out_shape)
  {
    TestIOGraph::init(in_shape, out_shape);

    _front_perm = g()->nodes()->create<luci::CircleConst>();
    {
      _front_perm->name("front_transpose/perm");
      init_circle_const(_front_perm, front_perm);
    }

    _front_transpose = g()->nodes()->create<luci::CircleTranspose>();
    {
      _front_transpose->a(input());
      _front_transpose->name("front_transpose");
      _front_transpose->perm(_front_perm);
    }

    _mid_shape = g()->nodes()->create<luci::CircleConst>();
    {
      _mid_shape->name("mid_reshpae/shape");
      init_circle_const(_mid_shape, mid_shape);
    }

    _mid_reshape = g()->nodes()->create<luci::CircleReshape>();
    {
      _mid_reshape->name("mid_reshape");
      _mid_reshape->tensor(_front_transpose);
      _mid_reshape->shape(_mid_shape);
    }

    _back_perm = g()->nodes()->create<luci::CircleConst>();
    {
      _back_perm->name("back_transpose/perm");
      init_circle_const(_back_perm, back_perm);
    }

    _back_transpose = g()->nodes()->create<luci::CircleTranspose>();
    {
      _back_transpose->name("back_transpose");
      _back_transpose->a(_mid_reshape);
      _back_transpose->perm(_back_perm);
    }

    output()->from(_back_transpose);
  }

  // create input-transpose-transpose-output graph
  void init_transpose_only(ShapeU32 in_shape, ShapeU32 front_perm, ShapeU32 back_perm,
                           ShapeU32 out_shape)
  {
    TestIOGraph::init(in_shape, out_shape);

    _front_perm = g()->nodes()->create<luci::CircleConst>();
    {
      _front_perm->name("front_transpose/perm");
      init_circle_const(_front_perm, front_perm);
    }

    _front_transpose = g()->nodes()->create<luci::CircleTranspose>();
    {
      _front_transpose->a(input());
      _front_transpose->name("front_transpose");
      _front_transpose->perm(_front_perm);
    }

    _back_perm = g()->nodes()->create<luci::CircleConst>();
    {
      _back_perm->name("back_transpose/perm");
      init_circle_const(_back_perm, back_perm);
    }

    _back_transpose = g()->nodes()->create<luci::CircleTranspose>();
    {
      _back_transpose->name("back_transpose");
      _back_transpose->a(_front_transpose);
      _back_transpose->perm(_back_perm);
    }

    output()->from(_back_transpose);
  }

private:
  void init_circle_const(luci::CircleConst *const_node, ShapeU32 shape)
  {
    const_node->dtype(loco::DataType::S32);
    const_node->size<loco::DataType::S32>(shape.size());
    uint32_t i = 0;
    for (auto v : shape)
    {
      const_node->at<loco::DataType::S32>(i++) = v;
    }
  }

  luci::CircleTranspose *_front_transpose = nullptr;
  luci::CircleConst *_front_perm = nullptr;

  luci::CircleReshape *_mid_reshape = nullptr;
  luci::CircleConst *_mid_shape = nullptr;

  luci::CircleTranspose *_back_transpose = nullptr;
  luci::CircleConst *_back_perm = nullptr;
};

bool is_transpose_removed(loco::Graph *g)
{
  bool transpose_exist = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (dynamic_cast<luci::CircleTranspose *>(node))
    {
      transpose_exist = true;
      break;
    }
  }
  return not transpose_exist;
}

} // namespace

TEST(RemoveUnnecessaryTransposeNetPass, rank_reduction_pattern1)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (1, 14, 14, 192)
   *      |
   * (1, 192, 14, 14)
   *      |
   * (1, 192, 196)
   *      |
   * (1, 196, 192)
   */
  g.init_whole_graph(/*in*/ {1, 14, 14, 192}, /*perm*/ {0, 3, 1, 2}, /*reshape*/ {1, 192, 196},
                     /*perm*/ {0, 2, 1}, /*out*/ {1, 196, 192});

  EXPECT_TRUE(pass.run(g.g()));
  EXPECT_TRUE(is_transpose_removed(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, rank_reduction_pattern2)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (1, 100, 10, 12)
   *        |
   * (1, 10, 12, 100)
   *        |
   *    (120, 100)
   *        |
   *    (100, 120)
   */
  g.init_whole_graph(/*in*/ {1, 100, 10, 12}, /*perm*/ {0, 2, 3, 1}, /*reshape*/ {120, 100},
                     /*perm*/ {1, 0},
                     /*out*/ {100, 120});

  EXPECT_TRUE(pass.run(g.g()));
  EXPECT_TRUE(is_transpose_removed(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, identity_pattern)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (1, 2, 3)
   *     |
   * (1, 2, 3)
   *      |
   * (1, 2, 3)
   *      |
   * (1, 2, 3)
   */
  g.init_whole_graph(/*in*/ {1, 2, 3}, /*perm*/ {0, 1, 2}, /*reshape*/ {1, 2, 3},
                     /*perm*/ {0, 1, 2},
                     /*out*/ {1, 2, 3});

  EXPECT_TRUE(pass.run(g.g()));
  EXPECT_TRUE(is_transpose_removed(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, basic_pattern1_NEG)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (1, 2, 4, 6)
   *      |
   * (1, 2, 6, 4)
   *      |
   * (1, 12, 4)
   *      |
   * (1, 4, 12)
   */
  g.init_whole_graph(/*in*/ {1, 2, 4, 6}, /*perm*/ {0, 1, 3, 2}, /*reshape*/ {1, 12, 4},
                     /*perm*/ {0, 2, 1},
                     /*out*/ {1, 4, 12});

  EXPECT_FALSE(pass.run(g.g()));
  EXPECT_FALSE(is_transpose_removed(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, basic_pattern2_NEG)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (15, 10, 10)
   *      |
   * (10, 10, 15)
   *      |
   * (1, 1, 1500)
   *      |
   * (1500, 1, 1)
   */
  g.init_whole_graph(/*in*/ {15, 10, 10}, /*perm*/ {1, 2, 0}, /*reshape*/ {1, 1, 1500},
                     /*perm*/ {2, 0, 1},
                     /*out*/ {1500, 1, 1});

  EXPECT_FALSE(pass.run(g.g()));
  EXPECT_FALSE(is_transpose_removed(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, basic_pattern3_NEG)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   *  (1, 2, 3, 4)
   *       |
   * perm (0, 3, 1, 2)
   *      |
   *   (1, 4, 2, 3)
   *      |
   *  perm (0, 2, 3, 1)
   *      |
   *   (1, 2, 3, 4)
   */
  g.init_transpose_only(/*in*/ {1, 2, 3, 4}, /*perm*/ {0, 3, 1, 2}, /*perm*/ {0, 2, 3, 1},
                        /*out*/ {1, 2, 3, 4});

  EXPECT_FALSE(pass.run(g.g()));
  EXPECT_FALSE(is_transpose_removed(g.g()));
}
