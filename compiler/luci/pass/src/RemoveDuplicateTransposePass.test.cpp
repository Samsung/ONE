/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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
#include "RemoveDuplicateTransposePassInternal.h"

#include <vector>

#include <gtest/gtest.h>

namespace
{

void setValue(luci::CircleConst *node, const std::vector<int> &v)
{
  node->dtype(loco::DataType::S32);
  node->size<loco::DataType::S32>(v.size());
  node->rank(1);
  node->dim(0).set(v.size());
  for (int i = 0; i < v.size(); ++i)
  {
    node->at<loco::DataType::S32>(i) = v[i];
  }
}

/**
 *  Type1
 *  BEFORE
 *         |
 *   [CircleInput]    [CircleConst]
 *           \              /
 *           [CircleTranspose]  [CircleConst]
 *                   \              /
 *                   [CircleTranspose]
 *                           |
 *
 *  AFTER
 *         |
 *   [CircleInput]
 *         |   Remove Both
 *
 */

class SimpleGraphType1
{
public:
  SimpleGraphType1()
  {
    input = g.nodes()->create<luci::CircleInput>();
    main_trans = g.nodes()->create<luci::CircleTranspose>();
    main_perm = g.nodes()->create<luci::CircleConst>();
    pred_trans = g.nodes()->create<luci::CircleTranspose>();
    pred_perm = g.nodes()->create<luci::CircleConst>();

    input->dtype(loco::DataType::FLOAT32);
    main_trans->dtype(loco::DataType::FLOAT32);
    main_perm->dtype(loco::DataType::S32);
    pred_trans->dtype(loco::DataType::FLOAT32);
    pred_perm->dtype(loco::DataType::S32);

    setValue(main_perm, {1, 0, 2, 3});
    setValue(pred_perm, {1, 0, 2, 3});

    main_trans->a(input);
    main_trans->perm(main_perm);
    pred_trans->a(main_trans);
    pred_trans->perm(pred_perm);
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleTranspose *pred_trans = nullptr;
  luci::CircleTranspose *main_trans = nullptr;
  luci::CircleConst *pred_perm = nullptr;
  luci::CircleConst *main_perm = nullptr;
};

/**
 *  Type2
 *  BEFORE
 *         |
 *   [CircleInput]    [CircleConst]
 *           \              /
 *           [CircleTranspose]  [CircleConst]
 *                   \               /
 *                   [CircleTranspose]
 *                           |
 *
 *  AFTER
 *          |                 |
 *    [CircleInput]     [CircleConst]
 *           \               /
 *           [CircleTranspose]
 *                   |
 *
 */

class SimpleGraphType2
{
public:
  SimpleGraphType2()
  {
    input = g.nodes()->create<luci::CircleInput>();
    main_trans = g.nodes()->create<luci::CircleTranspose>();
    main_perm = g.nodes()->create<luci::CircleConst>();
    pred_trans = g.nodes()->create<luci::CircleTranspose>();
    pred_perm = g.nodes()->create<luci::CircleConst>();

    input->dtype(loco::DataType::FLOAT32);
    main_trans->dtype(loco::DataType::FLOAT32);
    main_perm->dtype(loco::DataType::S32);
    pred_trans->dtype(loco::DataType::FLOAT32);
    pred_perm->dtype(loco::DataType::S32);

    setValue(main_perm, {0, 1, 3, 2});
    setValue(pred_perm, {1, 0, 2, 3});

    main_trans->a(input);
    main_trans->perm(main_perm);
    pred_trans->a(main_trans);
    pred_trans->perm(pred_perm);
  }

public:
  loco::Graph g;
  luci::CircleInput *input = nullptr;
  luci::CircleTranspose *pred_trans = nullptr;
  luci::CircleTranspose *main_trans = nullptr;
  luci::CircleConst *pred_perm = nullptr;
  luci::CircleConst *main_perm = nullptr;
};

} // namespace

TEST(RemoveDuplicateTransposePass, check_perm)
{
  luci::CircleConst const_node1, const_node2;

  setValue(&const_node1, {1, 0, 2, 3});
  setValue(&const_node2, {1, 0, 2, 3});
  EXPECT_TRUE(check_perm(&const_node1, &const_node2));
}

TEST(RemoveDuplicateTransposePass, check_perm_NEG)
{
  luci::CircleConst const_node1, const_node2;

  setValue(&const_node1, {0, 1, 3, 2});
  setValue(&const_node2, {1, 0, 2, 3});
  EXPECT_FALSE(check_perm(&const_node1, &const_node2));
}

TEST(RemoveDuplicateTransposePass, remove_duplicate_transpose_function_type1)
{
  SimpleGraphType1 g;
  EXPECT_TRUE(check_perm(g.pred_perm, g.main_perm));
  EXPECT_TRUE(remove_duplicate_transpose_function(g.pred_trans));
}

TEST(RemoveDuplicateTransposePass, remove_duplicate_transpose_function_type2)
{
  SimpleGraphType2 g;

  auto perm = loco::must_cast<luci::CircleConst *>(g.main_trans->perm());
  ASSERT_EQ(0, perm->at<loco::DataType::S32>(0));
  ASSERT_EQ(1, perm->at<loco::DataType::S32>(1));
  ASSERT_EQ(3, perm->at<loco::DataType::S32>(2));
  ASSERT_EQ(2, perm->at<loco::DataType::S32>(3));

  EXPECT_FALSE(check_perm(g.pred_perm, g.main_perm));
  EXPECT_TRUE(remove_duplicate_transpose_function(g.pred_trans));

  perm = loco::must_cast<luci::CircleConst *>(g.main_trans->perm());
  ASSERT_EQ(1, perm->at<loco::DataType::S32>(0));
  ASSERT_EQ(0, perm->at<loco::DataType::S32>(1));
  ASSERT_EQ(3, perm->at<loco::DataType::S32>(2));
  ASSERT_EQ(2, perm->at<loco::DataType::S32>(3));
}
