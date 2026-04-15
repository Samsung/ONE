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

#include "coco/IR/Use.h"
#include "coco/IR/ObjectManager.h"

#include "coco/IR/FeatureObject.h"

#include "Consumer.mock.h"

#include <memory>

#include <gtest/gtest.h>

using std::make_unique;

namespace
{
class UseTest : public ::testing::Test
{
protected:
  coco::ObjectManager obj_mgr;
};
} // namespace

TEST_F(UseTest, constructor)
{
  auto o = obj_mgr.create<coco::FeatureObject>();

  // TODO Rename 'use'
  ::mock::Consumer use;

  coco::Use slot{&use};

  ASSERT_EQ(slot.value(), nullptr);
}

TEST_F(UseTest, value)
{
  auto o = obj_mgr.create<coco::FeatureObject>();

  // TODO Rename 'use'
  ::mock::Consumer use;

  coco::Use slot{&use};

  slot.value(o);

  ASSERT_EQ(slot.value(), o);

  ASSERT_EQ(o->uses()->size(), 1);
  ASSERT_NE(o->uses()->find(&slot), o->uses()->end());

  slot.value(nullptr);

  ASSERT_EQ(slot.value(), nullptr);

  ASSERT_EQ(o->uses()->size(), 0);
}

TEST_F(UseTest, destructor)
{
  ::mock::Consumer consumer;

  auto o = obj_mgr.create<coco::FeatureObject>();
  auto use = make_unique<coco::Use>(&consumer);

  use->value(o);
  use.reset();

  // ~Use SHOULD unlink itself from linked Object (if exists)
  ASSERT_EQ(o->uses()->size(), 0);
}
