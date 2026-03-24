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

#include "coco/IR/Def.h"
#include "coco/IR/ObjectManager.h"

#include "coco/IR/FeatureObject.h"

#include <memory>

#include "Producer.mock.h"

#include <gtest/gtest.h>

using std::make_unique;

namespace
{
class DefTest : public ::testing::Test
{
protected:
  coco::ObjectManager obj_mgr;
};
} // namespace

TEST_F(DefTest, constructor)
{
  auto o = obj_mgr.create<coco::FeatureObject>();

  ::mock::Producer producer;
  coco::Def def{&producer};

  ASSERT_EQ(def.value(), nullptr);
}

TEST_F(DefTest, value)
{
  auto o = obj_mgr.create<coco::FeatureObject>();

  ::mock::Producer producer;
  coco::Def def{&producer};

  def.value(o);

  ASSERT_EQ(def.value(), o);

  ASSERT_EQ(o->def(), &def);

  def.value(nullptr);

  ASSERT_EQ(o->def(), nullptr);
}

TEST_F(DefTest, unlink_on_destruction)
{
  auto o = obj_mgr.create<coco::FeatureObject>();

  ::mock::Producer producer;
  auto def = make_unique<coco::Def>(&producer);

  def->value(o);
  ASSERT_EQ(o->def(), def.get());

  // Let's destruct the allocated slot
  def.reset(nullptr);

  // The def of Object SHOULD BE updated
  ASSERT_EQ(o->def(), nullptr);
}
