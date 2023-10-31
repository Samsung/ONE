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
  void init(ShapeU32 in_shape, ShapeU32 front_perm, ShapeU32 mid_shape, ShapeU32 back_perm,
            ShapeU32 out_shape)
  {
    TestIOGraph::init(in_shape, out_shape);

    auto init_circle_const = [](luci::CircleConst *const_node, ShapeU32 shape) {
      const_node->dtype(loco::DataType::S32);
      const_node->size<loco::DataType::S32>(shape.size());
      uint32_t i = 0;
      for (auto v : shape)
      {
        const_node->at<loco::DataType::S32>(i++) = v;
      }
    };

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

private:
  luci::CircleTranspose *_front_transpose;
  luci::CircleConst *_front_perm;

  luci::CircleReshape *_mid_reshape;
  luci::CircleConst *_mid_shape;

  luci::CircleTranspose *_back_transpose;
  luci::CircleConst *_back_perm;
};

} // namespace

TEST(RemoveUnnecessaryTransposeNetPass, rank_reduction_pattern1)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * from efficient-former
   *
   * (1, 14, 14, 192)
   *      A   B   C
   *      |
   * (1, 192, 14, 14)
   *      C   A   B
   *      |
   * (1, 192, 196)
   *      C   AB
   *      |
   * (1, 196, 192)
   *      AB   C
   */
  g.init(/*in*/ {1, 14, 14, 192}, /*perm*/ {0, 3, 1, 2}, /*reshape*/ {1, 192, 196},
         /*perm*/ {0, 2, 1}, /*out*/ {1, 196, 192});

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, rank_reduction_pattern2)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /** from detr
   *
   * (1, 7, 7, 448)
   *     A  B   C
   *      |
   * (1, 448, 7, 7)
   *      C   A  B
   *       |
   * (1, 448, 49)
   *      C   AB
   *      |
   * (1, 49, 448)
   *     AB   C
   */
  g.init(/*in*/ {1, 7, 7, 448}, /*perm*/ {0, 3, 1, 2}, /*reshape*/ {1, 448, 49}, /*perm*/ {0, 2, 1},
         /*out*/ {1, 49, 448});

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, rank_expand_pattern1)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (10, 15, 20)
   *  A   B   C
   *      |           => transpose
   * (10, 20, 15)
   *  A   C   B
   *      |           => reshape
   * (10, 20, 3, 5)
   *  A   C   Ba Bb
   *      |           => transpose
   * (10, 3, 5, 20)
   *  A   Ba Bb C
   */

  g.init(/*in*/ {10, 15, 20}, /*perm*/ {0, 2, 1}, /*reshape*/ {10, 20, 3, 5}, /*perm*/ {0, 2, 3, 1},
         /*out*/ {10, 3, 5, 20});

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, rank_expand_with_one_pattern)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (20, 50, 100)
   *      |           => transpose
   * (20, 100, 50)
   *      |           => reshape
   * (2, 1, 10, 10, 10, 5, 10, 1)
   *      |           => transpose
   * (1, 1, 2, 10, 5, 10, 10, 10)
   */

  g.init(/*in*/ {20, 50, 100}, /*perm*/ {0, 2, 1}, /*reshape*/ {2, 1, 10, 10, 10, 5, 10, 1},
         /*perm*/ {7, 1, 0, 2, 5, 6, 3, 4},
         /*out*/ {1, 1, 2, 10, 5, 10, 10, 10});

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, rank_reduction_with_one_pattern)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (1, 100, 10, 12)
   *      A   B   C
   *        |
   * (1, 10, 12, 100)
   *     B   C   A
   *        |
   *    (120, 100)
   *      BC   A
   *        |
   *    (100, 120)
   *      A   BC
   */
  g.init(/*in*/ {1, 100, 10, 12}, /*perm*/ {0, 2, 3, 1}, /*reshape*/ {120, 100}, /*perm*/ {1, 0},
         /*out*/ {100, 120});

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, complex_reshape_pattern1)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (12, 12, 15, 2)
   *   A  B   C   D
   *      |
   * (2, 12, 12, 15)
   *  D  A   B   C
   *      |
   * (2, 36, 12, 5)
   *  D  ABCa ABCb ABCc
   *      |
   * (36, 12, 5, 2)
   * ABCa ABCb ABCc D
   */

  g.init(/*in*/ {12, 12, 15, 2}, /*perm*/ {3, 0, 1, 2}, /*reshape*/ {2, 36, 12, 5},
         /*perm*/ {1, 2, 3, 0},
         /*out*/ {100, 120});

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, complex_reshape_pattern2)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (02, 10, 20, 30)
   *        |
   * (10, 20, 30, 02)
   *        |
   * (20, 05, 60, 02)
   *        |
   * (02, 20, 05, 60)
   */

  g.init(/*in*/ {2, 10, 20, 30}, /*perm*/ {1, 2, 3, 0}, /*reshape*/ {20, 5, 60, 2},
         /*perm*/ {3, 0, 1, 2},
         /*out*/ {2, 20, 5, 60});

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, complex_reshape_pattern3)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (1, 100, 200, 100, 1)
   *        |
   * (1, 200, 100, 100, 1)
   *        |
   * (20, 10, 100, 10, 10)
   *        |
   * (100, 20, 10, 10, 10)
   */

  g.init(/*in*/ {1, 100, 200, 100, 1}, /*perm*/ {0, 2, 1, 3, 4}, /*reshape*/ {20, 10, 100, 10, 10},
         /*perm*/ {2, 0, 1, 3, 4},
         /*out*/ {100, 20, 10, 10, 10});

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, one_rank_pattern)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (1)
   *  |
   * (1)
   *  |
   * (1)
   *  |
   * (1)
   */

  g.init(/*in*/ {1}, /*perm*/ {0}, /*reshape*/ {1}, /*perm*/ {0}, /*out*/ {11});

  EXPECT_TRUE(pass.run(g.g()));
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
  g.init(/*in*/ {1, 2, 3}, /*perm*/ {0, 1, 2}, /*reshape*/ {1, 2, 3}, /*perm*/ {0, 1, 2},
         /*out*/ {1, 2, 3});

  EXPECT_TRUE(pass.run(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, basic_pattern_NEG)
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
  g.init(/*in*/ {1, 2, 4, 6}, /*perm*/ {0, 1, 3, 2}, /*reshape*/ {1, 12, 4}, /*perm*/ {0, 2, 1},
         /*out*/ {1, 4, 12});

  EXPECT_FALSE(pass.run(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, fuse_after_reordered_pattern_NEG)
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
  g.init(/*in*/ {15, 10, 10}, /*perm*/ {1, 2, 0}, /*reshape*/ {1, 1, 1500}, /*perm*/ {2, 0, 1},
         /*out*/ {1500, 1, 1});

  EXPECT_FALSE(pass.run(g.g()));
}

TEST(RemoveUnnecessaryTransposeNetPass, split_after_reorderd_pattern_NEG)
{
  TransposeReshapeTransposeGraph g;
  luci::RemoveUnnecessaryTransposeNetPass pass;

  /**
   * (10, 12, 60)
   *      |
   * (60, 10, 12)
   *      |
   * (03, 04, 05, 10, 12)
   *      |
   * (10, 12, 04, 03, 05)
   */
  g.init(/*in*/ {10, 12, 60}, /*perm*/ {2, 0, 1}, /*reshape*/ {3, 4, 5, 10, 12},
         /*perm*/ {3, 4, 1, 0, 2},
         /*out*/ {10, 12, 4, 3, 5});

  EXPECT_FALSE(pass.run(g.g()));
}
