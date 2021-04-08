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

#include "luci/Service/Nodes/CircleConst.h"

#include <loco.h>
#include <loco/IR/Graph.h>

#include <gtest/gtest.h>

TEST(CircleConstTest, clone)
{
  auto g = loco::make_graph();

  // prepare source CircleConst
  auto circle_const = g->nodes()->create<luci::CircleConst>();

  const auto size = 2;

  circle_const->dtype(loco::DataType::S32);
  circle_const->rank(1);
  circle_const->dim(0).set(size);
  circle_const->shape_status(luci::ShapeStatus::VALID);

  circle_const->size<loco::DataType::S32>(size);
  for (uint32_t i = 0; i < size; i++)
    circle_const->at<loco::DataType::S32>(i) = i;

  // quantparam
  auto quantparam = std::make_unique<luci::CircleQuantParam>();
  quantparam->scale = {1.0};
  quantparam->zerop = {0};
  quantparam->min = {-127.0};
  quantparam->max = {127.0};
  quantparam->quantized_dimension = 1;
  circle_const->quantparam(std::move(quantparam));

  // sparsityparam
  auto sparam = std::make_unique<luci::SparsityParam>();
  sparam->traversal_order = {1};
  sparam->block_map = {1};
  sparam->dim_metadata = {};
  circle_const->sparsityparam(std::move(sparam));

  // make a clone
  auto const_cloned = luci::clone(circle_const);

  // check attributes
  ASSERT_EQ(loco::DataType::S32, const_cloned->dtype());
  ASSERT_EQ(1, const_cloned->rank());
  ASSERT_EQ(2, const_cloned->dim(0).value());
  ASSERT_EQ(2, const_cloned->size<loco::DataType::S32>());
  ASSERT_EQ(0, const_cloned->at<loco::DataType::S32>(0));
  ASSERT_EQ(1, const_cloned->at<loco::DataType::S32>(1));
  ASSERT_NE(nullptr, const_cloned->quantparam());
  ASSERT_NE(nullptr, const_cloned->sparsityparam());
}

TEST(CircleConstTest, clone_U8)
{
  auto g = loco::make_graph();

  // prepare source CircleConst
  auto circle_const = g->nodes()->create<luci::CircleConst>();

  const auto size = 0;

  circle_const->dtype(loco::DataType::U8);
  circle_const->rank(1);
  circle_const->dim(0).set(size);
  circle_const->shape_status(luci::ShapeStatus::VALID);
  circle_const->size<loco::DataType::U8>(size);

  // make a clone
  auto const_cloned = luci::clone(circle_const);

  // check attributes
  ASSERT_EQ(loco::DataType::U8, const_cloned->dtype());
}

TEST(CircleConstTest, clone_S8)
{
  auto g = loco::make_graph();

  // prepare source CircleConst
  auto circle_const = g->nodes()->create<luci::CircleConst>();

  const auto size = 0;

  circle_const->dtype(loco::DataType::S8);
  circle_const->rank(1);
  circle_const->dim(0).set(size);
  circle_const->shape_status(luci::ShapeStatus::VALID);
  circle_const->size<loco::DataType::S8>(size);

  // make a clone
  auto const_cloned = luci::clone(circle_const);

  // check attributes
  ASSERT_EQ(loco::DataType::S8, const_cloned->dtype());
}

TEST(CircleConstTest, clone_S64)
{
  auto g = loco::make_graph();

  // prepare source CircleConst
  auto circle_const = g->nodes()->create<luci::CircleConst>();

  const auto size = 0;

  circle_const->dtype(loco::DataType::S64);
  circle_const->rank(1);
  circle_const->dim(0).set(size);
  circle_const->shape_status(luci::ShapeStatus::VALID);
  circle_const->size<loco::DataType::S64>(size);

  // make a clone
  auto const_cloned = luci::clone(circle_const);

  // check attributes
  ASSERT_EQ(loco::DataType::S64, const_cloned->dtype());
}

TEST(CircleConstTest, clone_BOOL)
{
  auto g = loco::make_graph();

  // prepare source CircleConst
  auto circle_const = g->nodes()->create<luci::CircleConst>();

  const auto size = 0;

  circle_const->dtype(loco::DataType::BOOL);
  circle_const->rank(1);
  circle_const->dim(0).set(size);
  circle_const->shape_status(luci::ShapeStatus::VALID);
  circle_const->size<loco::DataType::BOOL>(size);

  // make a clone
  auto const_cloned = luci::clone(circle_const);

  // check attributes
  ASSERT_EQ(loco::DataType::BOOL, const_cloned->dtype());
}
