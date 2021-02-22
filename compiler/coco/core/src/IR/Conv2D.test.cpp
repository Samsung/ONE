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

#include "coco/IR/Ops.h"
#include "coco/IR/ObjectManager.h"

#include <vector>
#include <memory>

#include <gtest/gtest.h>

using std::make_unique;

namespace
{
class Conv2DTest : public ::testing::Test
{
public:
  Conv2DTest()
  {
    // DO NOTHING
  }

protected:
  coco::Conv2D *allocate(void)
  {
    auto op = new coco::Conv2D;
    _allocated.emplace_back(op);
    return op;
  }

protected:
  coco::ObjectManager obj_mgr;

private:
  std::vector<std::unique_ptr<coco::Conv2D>> _allocated;
};
} // namespace

TEST_F(Conv2DTest, ctor)
{
  auto op = allocate();

  // arg() should be initialized as nullptr on construction
  ASSERT_EQ(op->arg(), nullptr);
  // ker() should be initialized as nullptr on construction
  ASSERT_EQ(op->ker(), nullptr);

  // uses() should be empty on construction
  ASSERT_EQ(op->uses().size(), 0);
  // parent() should be nullptr on construction
  ASSERT_EQ(op->parent(), nullptr);

  ASSERT_EQ(op->group(), 1);

  ASSERT_NE(op->pad(), nullptr);
  ASSERT_EQ(op->pad()->top(), 0);
  ASSERT_EQ(op->pad()->bottom(), 0);
  ASSERT_EQ(op->pad()->left(), 0);
  ASSERT_EQ(op->pad()->right(), 0);

  ASSERT_NE(op->stride(), nullptr);
  ASSERT_EQ(op->stride()->vertical(), 1);
  ASSERT_EQ(op->stride()->horizontal(), 1);
}

TEST_F(Conv2DTest, asConv2D)
{
  auto op = allocate();

  coco::Op *mutable_base = op;
  const coco::Op *immutable_base = op;

  ASSERT_EQ(mutable_base->asConv2D(), op);
  ASSERT_EQ(mutable_base->asConv2D(), immutable_base->asConv2D());
}

namespace
{
struct IsConv2D : public coco::Op::Visitor<bool>
{
  bool visit(const coco::Conv2D *) override { return true; }
};
} // namespace

TEST_F(Conv2DTest, ker_update)
{
  // Prepare a kernel object for testing
  auto obj = obj_mgr.create<coco::KernelObject>();

  // Test 'Conv2D' class
  auto op = allocate();

  op->ker(obj);
  ASSERT_EQ(op->ker(), obj);

  // Op now uses 'obj'
  {
    auto uses = op->uses();

    ASSERT_NE(uses.find(obj), uses.end());
  }

  // ker method should enlist op itself as a consumer of a given kernel object
  {
    auto consumers = coco::consumers(obj);

    ASSERT_EQ(consumers.size(), 1);
    ASSERT_NE(consumers.find(op), consumers.end());
  }
}

TEST_F(Conv2DTest, accept)
{
  // Test 'Conv2D' class
  auto op = allocate();

  coco::Conv2D *mutable_ptr = op;
  const coco::Conv2D *immutable_ptr = op;

  ASSERT_TRUE(mutable_ptr->accept(IsConv2D{}));
  ASSERT_TRUE(immutable_ptr->accept(IsConv2D{}));
}

TEST_F(Conv2DTest, destructor)
{
  // Prepare a kernel object for testing
  auto obj = obj_mgr.create<coco::KernelObject>();

  // Create 'Conv2D' op
  auto op = make_unique<coco::Conv2D>();

  op->ker(obj);

  // Destroy 'Conv2D' op
  op.reset();

  ASSERT_EQ(obj->uses()->size(), 0);
}
