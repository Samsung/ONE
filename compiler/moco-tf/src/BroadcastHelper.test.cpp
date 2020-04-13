/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "BroadcastHelper.h"

#include <loco.h>

#include <gtest/gtest.h>

TEST(BroadcastFunctorTest, expand_rank)
{
  // Broadcast Tensor<3> as Tensor<1 x 3>
  auto g = loco::make_graph();

  auto input = g->inputs()->create();

  auto pull = g->nodes()->create<loco::Pull>();
  pull->index(0);

  loco::TensorShape current_shape;
  {
    current_shape.rank(1);
    current_shape.dim(0) = 3;
  }

  loco::TensorShape expected_shape;
  {
    expected_shape.rank(2);
    expected_shape.dim(0) = 1;
    expected_shape.dim(1) = 3;
  }

  moco::tf::BroadcastFunctor functor{expected_shape};

  auto node = functor.build(pull, current_shape);

  ASSERT_EQ(node->opnum(), static_cast<uint32_t>(loco::CanonicalOpcode::FixedReshape));
  ASSERT_EQ(node->arg(0), pull);
}

TEST(BroadcastFunctorTest, expand_dims)
{
  // Broadcast Tensor<1> as Tensor<3>
  auto g = loco::make_graph();

  auto input = g->inputs()->create();

  auto pull = g->nodes()->create<loco::Pull>();
  pull->index(0);

  loco::TensorShape current_shape;
  {
    current_shape.rank(1);
    current_shape.dim(0) = 1;
  }

  loco::TensorShape expected_shape;
  {
    expected_shape.rank(1);
    expected_shape.dim(0) = 3;
  }

  moco::tf::BroadcastFunctor functor{expected_shape};

  auto node = functor.build(pull, current_shape);

  ASSERT_EQ(node->opnum(), static_cast<uint32_t>(loco::CanonicalOpcode::TensorBroadcast));
  ASSERT_EQ(node->arg(0), pull);

  auto tensor_broadcast = dynamic_cast<loco::TensorBroadcast *>(node);

  ASSERT_NE(tensor_broadcast, nullptr);
  ASSERT_TRUE(tensor_broadcast->mapping()->defined(0));
  ASSERT_EQ(tensor_broadcast->mapping()->dim(0), 3);
}
