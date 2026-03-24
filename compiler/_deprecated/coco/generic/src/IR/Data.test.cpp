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

#include "coco/IR/Data.h"
#include "coco/IR/Module.h"
#include "coco/IR/KernelLayouts.h"

#include <nncc/core/ADT/kernel/NCHWLayout.h>

#include <gtest/gtest.h>

TEST(IR_DATA, construct)
{
  auto data = coco::Data::create();

  coco::Data *mutable_ptr = data.get();
  const coco::Data *immutable_ptr = data.get();

  ASSERT_NE(mutable_ptr->f32(), nullptr);
  ASSERT_EQ(mutable_ptr->f32(), immutable_ptr->f32());
}

TEST(IR_DATA, allocate_and_link_bag)
{
  auto m = coco::Module::create();
  auto d = coco::Data::create();

  // Create a bag
  auto bag = m->entity()->bag()->create(9);

  // weight(...) SHOULD return a null-span for an invalid bag
  {
    auto span = d->f32()->weight(bag);

    ASSERT_EQ(span.data(), nullptr);
    ASSERT_EQ(span.size(), 0);
  }

  // Allocate a weight space
  {
    auto allocated_span = d->f32()->allocate(bag);

    ASSERT_NE(allocated_span.data(), nullptr);
    ASSERT_EQ(allocated_span.size(), bag->size());

    auto retrieved_span = d->f32()->weight(bag);

    ASSERT_EQ(allocated_span.data(), retrieved_span.data());
    ASSERT_EQ(allocated_span.size(), retrieved_span.size());
  }
}
