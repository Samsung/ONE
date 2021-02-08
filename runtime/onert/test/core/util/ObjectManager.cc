/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include "util/ObjectManager.h"
#include "util/Index.h"

using namespace onert;

struct TestTag;
using Index = typename util::Index<uint32_t, TestTag>;

TEST(ObjectManager, emplace)
{
  util::ObjectManager<Index, int> man;

  auto index = man.emplace(100);
  ASSERT_EQ(man.at(index), 100);
}

TEST(ObjectManager, neg_remove_1)
{
  util::ObjectManager<Index, int> man;

  Index index = man.emplace(100);
  ASSERT_TRUE(man.exist(index));
  ASSERT_EQ(man.at(index), 100);

  man.remove(index);
  ASSERT_FALSE(man.exist(index));
}

TEST(ObjectManager, neg_remove_2)
{
  util::ObjectManager<Index, int> man;

  auto index0 = man.emplace(100);
  auto index1 = man.emplace(200);
  ASSERT_TRUE(man.exist(index0));
  ASSERT_EQ(man.at(index0), 100);
  ASSERT_TRUE(man.exist(index1));
  ASSERT_EQ(man.at(index1), 200);

  man.remove(index0);
  ASSERT_FALSE(man.exist(index0));
  ASSERT_TRUE(man.exist(index1));
  ASSERT_EQ(man.at(index1), 200);
}

TEST(ObjectManager, push)
{
  util::ObjectManager<Index, int> man;

  // Not specify index
  auto index = man.push(std::make_unique<int>(100));
  ASSERT_EQ(man.at(index), 100);

  // Specify index
  auto index2 = man.push(std::make_unique<int>(200), Index{33});
  ASSERT_EQ(index2.value(), 33);
  ASSERT_EQ(man.at(index2), 200);

  auto index3 = man.push(std::make_unique<int>(300));
  // NOTE auto-generated index number is always (biggest index in the ObjectManager + 1)
  ASSERT_EQ(index3.value(), 34);
  ASSERT_EQ(man.at(index3), 300);

  auto index4 = man.push(std::make_unique<int>(400), Index{22});
  ASSERT_EQ(index4.value(), 22);
  ASSERT_EQ(man.at(index4), 400);

  auto index5 = man.push(std::make_unique<int>(500));
  // NOTE auto-generated index number is always (biggest index in the ObjectManager + 1)
  ASSERT_EQ(index5.value(), 35);
  ASSERT_EQ(man.at(index5), 500);
}

TEST(ObjectManager, neg_push)
{
  util::ObjectManager<Index, int> man;

  // Specify index
  auto index = man.push(std::make_unique<int>(100), Index{55});
  ASSERT_EQ(index.value(), 55);
  ASSERT_EQ(man.at(index), 100);

  // Specify the same index
  auto index2 = man.push(std::make_unique<int>(200), Index{55});
  ASSERT_FALSE(index2.valid());
}

static const uint32_t kMaxUInt32 = std::numeric_limits<uint32_t>::max();

TEST(ObjectManager, neg_push_undefined_index)
{
  util::ObjectManager<Index, int> man;

  // Try inserting invalid(undefined) index
  auto index = man.push(std::make_unique<int>(100), Index{kMaxUInt32});
  ASSERT_FALSE(index.valid());
  ASSERT_EQ(man.size(), 0);
}

TEST(ObjectManager, neg_push_max_index)
{
  util::ObjectManager<Index, int> man;

  // Insert an object with maximum valid index
  auto index = man.push(std::make_unique<int>(100), Index{kMaxUInt32 - 1});
  ASSERT_EQ(index.value(), kMaxUInt32 - 1);
  ASSERT_EQ(man.at(index), 100);
  ASSERT_EQ(man.size(), 1);

  // Reached to the final index so next push/emplace must fail
  auto index2 = man.push(std::make_unique<int>(200));
  ASSERT_EQ(man.size(), 1);
  ASSERT_FALSE(index2.valid());
}

TEST(ObjectManager, neg_emplace_max_index)
{
  util::ObjectManager<Index, int> man;

  // Insert an object with maximum valid index
  auto index = man.push(std::make_unique<int>(100), Index{kMaxUInt32 - 1});
  ASSERT_EQ(index.value(), kMaxUInt32 - 1);
  ASSERT_EQ(man.at(index), 100);
  ASSERT_EQ(man.size(), 1);

  // Reached to the final index so next push/emplace must fail
  auto index3 = man.emplace(200);
  ASSERT_EQ(man.size(), 1);
  ASSERT_FALSE(index3.valid());
}

TEST(ObjectManager, const_iterate)
{
  util::ObjectManager<Index, int> man;

  auto index0 = man.emplace(100);
  auto index1 = man.emplace(200);
  auto index2 = man.emplace(300);

  int sum = 0;
  man.iterate([&](const Index &index, const int &val) { sum += val; });
  ASSERT_EQ(sum, 600);
}

TEST(ObjectManager, non_const_iterate)
{
  util::ObjectManager<Index, int> man;

  auto index0 = man.emplace(100);
  auto index1 = man.emplace(200);
  auto index2 = man.emplace(300);

  man.iterate([&](const Index &index, int &val) { val += 1; });
  ASSERT_EQ(man.at(index0), 101);
  ASSERT_EQ(man.at(index1), 201);
  ASSERT_EQ(man.at(index2), 301);
}

TEST(ObjectManager, set)
{
  util::ObjectManager<Index, int> man;
  auto index = man.set(Index{1}, std::make_unique<int>(100)); // Insert
  ASSERT_EQ(index, Index{1});
  auto index2 = man.set(index, std::make_unique<int>(200)); // Overwrite
  ASSERT_EQ(index2, index);
  ASSERT_EQ(man.at(index2), 200);
}

TEST(ObjectManager, neg_set)
{
  auto v = std::make_unique<int>(100);
  util::ObjectManager<Index, int> man;
  auto index = man.set(Index{}, std::move(v)); // Try set with an invalid index
  ASSERT_EQ(index, Index{});
  ASSERT_FALSE(index.valid());
  ASSERT_NE(v, nullptr); // v must be kept when failure
}

TEST(ObjectManager, getRawPtr)
{
  auto v = std::make_unique<int>(100);
  auto v_ptr = v.get();
  util::ObjectManager<Index, int> man;
  auto index = man.push(std::move(v));
  ASSERT_EQ(v_ptr, man.getRawPtr(index));
}

TEST(ObjectManager, neg_getRawPtr)
{
  util::ObjectManager<Index, int> man;
  auto ptr = man.getRawPtr(Index{1});
  ASSERT_EQ(ptr, nullptr);
}
