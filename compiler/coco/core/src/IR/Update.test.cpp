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

#include "coco/IR/Update.h"
#include "coco/IR/BagManager.h"

#include "Updater.mock.h"

#include <gtest/gtest.h>

namespace
{
class UpdateTest : public ::testing::Test
{
protected:
  coco::BagManager bag_mgr;
};
} // namespace

TEST_F(UpdateTest, constructor)
{
  // TODO Rename 'update'
  ::mock::Updater update;

  // TODO Rename 'slot'
  coco::Update slot{&update};

  ASSERT_EQ(slot.bag(), nullptr);
}

TEST_F(UpdateTest, value)
{
  // TODO Rename 'update'
  ::mock::Updater update;

  // TODO Rename 'slot'
  coco::Update slot{&update};

  auto bag = bag_mgr.create(16);

  slot.bag(bag);

  ASSERT_EQ(slot.bag(), bag);

  ASSERT_EQ(bag->updates()->size(), 1);
  ASSERT_NE(bag->updates()->find(&slot), bag->updates()->end());

  slot.bag(nullptr);

  ASSERT_EQ(slot.bag(), nullptr);

  ASSERT_EQ(bag->updates()->size(), 0);
}

TEST_F(UpdateTest, unlink_on_destruction)
{
  ::mock::Updater updater;

  auto bag = bag_mgr.create(1);

  {
    coco::Update update{&updater};
    update.bag(bag);
    ASSERT_EQ(bag->updates()->size(), 1);
  }

  ASSERT_EQ(bag->updates()->size(), 0);
}
