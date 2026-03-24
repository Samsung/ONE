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

#include "coco/ADT/PtrList.h"

#include <memory>

#include <gtest/gtest.h>

namespace
{
struct Object
{
};
} // namespace

TEST(ADT_PTR_LIST, ctor)
{
  coco::PtrList<Object> l;

  ASSERT_EQ(l.size(), 0);
}

TEST(ADT_PTR_LIST, insert)
{
  coco::PtrList<Object> l;

  std::unique_ptr<Object> ptr{new Object};

  l.insert(ptr.get());

  ASSERT_EQ(l.size(), 1);
  ASSERT_EQ(l.at(0), ptr.get());
}
