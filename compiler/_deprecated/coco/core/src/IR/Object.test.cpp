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

#include "coco/IR/Object.h"
#include "coco/IR/BagManager.h"

#include <vector>

#include <gtest/gtest.h>

namespace
{
class ObjectTest : public ::testing::Test
{
protected:
  coco::BagManager bag_mgr;
};
} // namespace

namespace
{
namespace mock
{
struct Object : public coco::Object
{
public:
  virtual ~Object() = default;
};
} // namespace mock
} // namespace

TEST_F(ObjectTest, ctor)
{
  ::mock::Object obj;

  // Newly created object should not have a backing bag
  ASSERT_EQ(obj.bag(), nullptr);

  // Newly created object should not have def and uses
  ASSERT_EQ(obj.def(), nullptr);
  ASSERT_TRUE(obj.uses()->empty());
}

TEST_F(ObjectTest, bag_update)
{
  // Prepare bag
  auto bag = bag_mgr.create(1);

  // Test 'Object' class through a mock-up object
  ::mock::Object obj;

  obj.bag(bag);

  // 'bag(Bag *)' should affect the return of 'bag(void)'
  ASSERT_EQ(obj.bag(), bag);

  // User SHOULD be able to access dependent objects through 'bag'
  {
    auto deps = coco::dependent_objects(bag);
    ASSERT_EQ(deps.size(), 1);
    ASSERT_EQ(deps.count(&obj), 1);
  }

  // Unlink Object-Bag relation
  obj.bag(nullptr);

  ASSERT_EQ(obj.bag(), nullptr);

  {
    auto deps = coco::dependent_objects(bag);
    ASSERT_EQ(deps.size(), 0);
  }
}

TEST_F(ObjectTest, destructor)
{
  auto bag = bag_mgr.create(1);

  // Destruct Object after proper initialization
  {
    ::mock::Object obj;

    obj.bag(bag);
  }

  // Object SHOULD be unlinked from Bag on destruction
  {
    auto deps = coco::dependent_objects(bag);
    ASSERT_EQ(deps.size(), 0);
  }
}

TEST_F(ObjectTest, safe_cast)
{
  ASSERT_EQ(coco::safe_cast<coco::FeatureObject>(nullptr), nullptr);
  ASSERT_EQ(coco::safe_cast<coco::KernelObject>(nullptr), nullptr);
}
