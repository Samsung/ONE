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

#include "stdex/Memory.h"

#include <gtest/gtest.h>

namespace
{

struct Stat
{
  unsigned allocated = 0;
  unsigned freed = 0;
};

struct Counter
{
public:
  Counter(Stat *stat) : _stat{stat} { _stat->allocated += 1; }

public:
  ~Counter() { _stat->freed += 1; }

private:
  Stat *_stat;
};

} // namespace

TEST(MemoryTest, make_unique)
{
  Stat stat;

  ASSERT_EQ(stat.allocated, 0);
  ASSERT_EQ(stat.freed, 0);

  auto o = stdex::make_unique<::Counter>(&stat);

  ASSERT_EQ(stat.allocated, 1);
  ASSERT_EQ(stat.freed, 0);

  o.reset();

  ASSERT_EQ(stat.allocated, 1);
  ASSERT_EQ(stat.freed, 1);
}
