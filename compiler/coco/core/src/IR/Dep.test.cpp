/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "coco/IR/Dep.h"

#include "coco/IR/BagManager.h"

#include "coco/IR/ObjectManager.h"
#include "coco/IR/FeatureObject.h"

#include <gtest/gtest.h>

using namespace nncc::core::ADT;

namespace
{
class DepTest : public ::testing::Test
{
protected:
  coco::BagManager bag_mgr;
  coco::ObjectManager obj_mgr;
};
} // namespace

TEST_F(DepTest, default_constructor)
{
  coco::Dep dep;

  ASSERT_EQ(dep.bag(), nullptr);
  ASSERT_EQ(dep.object(), nullptr);
}

TEST_F(DepTest, bag_update)
{
  auto bag = bag_mgr.create(3);

  coco::Dep dep;

  // NOTE b->object() is not updated here
  dep.bag(bag);

  ASSERT_EQ(dep.bag(), bag);
}

TEST_F(DepTest, bag_update_with_link_and_object)
{
  auto bag = bag_mgr.create(3);
  auto obj = obj_mgr.create<coco::FeatureObject>();

  coco::Dep dep;

  dep.object(obj);

  dep.bag(bag);

  auto deps = coco::dependent_objects(bag);

  ASSERT_EQ(deps.size(), 1);
  ASSERT_NE(deps.count(obj), 0);
}
