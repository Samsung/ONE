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

#include "coco/ADT/PtrManager.h"

#include <memory>

#include <gtest/gtest.h>

namespace
{
struct Count
{
  uint32_t allocated;
  uint32_t freed;

  Count() : allocated{0}, freed{0}
  {
    // DO NOTHING
  }
};

class Object
{
public:
  Object(Count *count, uint32_t value) : _count{count}, _value{value} { _count->allocated += 1; }

public:
  ~Object() { _count->freed += 1; }

public:
  uint32_t value(void) const { return _value; }

private:
  Count *const _count;

private:
  uint32_t _value;
};

struct ObjectManager final : public coco::PtrManager<Object>
{
  Object *alloc(Count *count, uint32_t value)
  {
    std::unique_ptr<Object> o{new Object{count, value}};
    return take(std::move(o));
  }

  void free(Object *o) { release(o); }
};
} // namespace

TEST(ADT_PTR_MANAGER, usecase)
{
  Count c;

  ASSERT_EQ(c.allocated, 0);
  ASSERT_EQ(c.freed, 0);

  {
    ::ObjectManager mgr;

    auto obj_1 = mgr.alloc(&c, 3);
    auto obj_2 = mgr.alloc(&c, 4);

    EXPECT_EQ(c.allocated, 2);
    ASSERT_EQ(c.freed, 0);

    EXPECT_EQ(mgr.size(), 2);
    EXPECT_EQ(mgr.at(0), obj_1);
    EXPECT_EQ(mgr.at(1), obj_2);

    // Let's delete obj_1
    mgr.free(obj_1);

    EXPECT_EQ(c.allocated, 2);
    ASSERT_EQ(c.freed, 1);

    EXPECT_EQ(mgr.size(), 1);
    EXPECT_EQ(mgr.at(0), obj_2);
  }

  // PtrManger SHOULD destruct all of the allocated object when it is destructed.
  ASSERT_EQ(c.allocated, 2);
  ASSERT_EQ(c.freed, 2);
}
