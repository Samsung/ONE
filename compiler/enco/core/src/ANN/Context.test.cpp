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

#include "Context.h"

#include <set>

#include <gtest/gtest.h>

namespace
{
class ANNContextTest : public ::testing::Test
{
public:
  ANNContextTest() { m = coco::Module::create(); }

public:
  virtual ~ANNContextTest() = default;

protected:
  std::unique_ptr<coco::Module> m;
};
} // namespace

TEST_F(ANNContextTest, constructor)
{
  ANNContext ann_ctx;

  ASSERT_EQ(ann_ctx.count(), 0);
}

TEST_F(ANNContextTest, create)
{
  ANNContext ann_ctx;

  auto blk = m->entity()->block()->create();
  auto binder = ann_ctx.create(blk);

  ASSERT_NE(binder, nullptr);
}

TEST_F(ANNContextTest, find)
{
  ANNContext ann_ctx;

  // CASE: Corresponding binder does not exist
  {
    auto blk = m->entity()->block()->create();
    ASSERT_EQ(ann_ctx.find(blk), nullptr);
  }

  // CASE: Corresponding binder does exist
  {
    auto blk = m->entity()->block()->create();
    auto binder_created = ann_ctx.create(blk);
    auto binder_found = ann_ctx.find(blk);

    ASSERT_EQ(binder_created, binder_found);
  }
}
