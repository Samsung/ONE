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

#include "coco/IR/Read.h"
#include "coco/IR/BagManager.h"

#include "Reader.mock.h"

#include <gtest/gtest.h>

namespace
{
class ReadTest : public ::testing::Test
{
protected:
  coco::BagManager bag_mgr;
};
} // namespace

TEST_F(ReadTest, constructor)
{
  // TODO Rename 'read' as 'reader'
  ::mock::Reader read;

  // TODO Rename 'slot'
  coco::Read slot{&read};

  ASSERT_EQ(slot.bag(), nullptr);
}

TEST_F(ReadTest, value)
{
  // TODO Rename 'read' as 'reader'
  ::mock::Reader read;

  // TODO Rename 'slot'
  coco::Read slot{&read};

  auto bag = bag_mgr.create(16);

  slot.bag(bag);

  ASSERT_EQ(slot.bag(), bag);

  ASSERT_EQ(bag->reads()->size(), 1);
  ASSERT_NE(bag->reads()->find(&slot), bag->reads()->end());

  slot.bag(nullptr);

  ASSERT_EQ(slot.bag(), nullptr);

  ASSERT_EQ(bag->reads()->size(), 0);
}

TEST_F(ReadTest, unlink_on_destruction)
{
  // TODO Rename 'read' as 'reader'
  ::mock::Reader reader;

  auto bag = bag_mgr.create(1);

  {
    coco::Read read{&reader};
    read.bag(bag);
  }

  ASSERT_EQ(bag->reads()->size(), 0);
}
