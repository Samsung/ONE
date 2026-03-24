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

#include "coco/IR/KernelObject.h"

#include <vector>
#include <memory>

#include <gtest/gtest.h>

using namespace nncc::core::ADT;

namespace
{
class KernelObjectTest : public ::testing::Test
{
protected:
  coco::KernelObject *allocate()
  {
    auto o = new coco::KernelObject{};
    _allocated.emplace_back(o);
    return o;
  }

  coco::KernelObject *allocate(const kernel::Shape &shape)
  {
    auto o = new coco::KernelObject{shape};
    _allocated.emplace_back(o);
    return o;
  }

private:
  std::vector<std::unique_ptr<coco::KernelObject>> _allocated;
};
} // namespace

TEST_F(KernelObjectTest, constructor)
{
  const nncc::core::ADT::kernel::Shape shape{1, 1, 3, 3};
  auto o = allocate(shape);

  ASSERT_EQ(o->shape(), shape);
  ASSERT_EQ(o->kind(), coco::Object::Kind::Kernel);
}

TEST_F(KernelObjectTest, asKernel)
{
  const nncc::core::ADT::kernel::Shape shape{1, 1, 3, 3};
  auto o = allocate(shape);

  coco::Object *mutable_object = o;
  const coco::Object *immutable_object = o;

  ASSERT_NE(mutable_object->asKernel(), nullptr);
  ASSERT_EQ(mutable_object->asKernel(), immutable_object->asKernel());
}

TEST_F(KernelObjectTest, casting_helpers)
{
  auto obj = allocate();

  ASSERT_TRUE(coco::isa<coco::KernelObject>(obj));
  ASSERT_EQ(coco::cast<coco::KernelObject>(obj), obj);
  ASSERT_EQ(coco::safe_cast<coco::KernelObject>(obj), obj);
}
