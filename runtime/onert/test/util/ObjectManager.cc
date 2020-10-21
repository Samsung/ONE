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

  auto index = man.push(std::unique_ptr<int>{new int{100}});
  ASSERT_EQ(man.at(index), 100);
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
