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

#include "coco/IR/FeatureObject.h"
#include "coco/IR/FeatureLayouts.h"

#include <vector>
#include <memory>

#include <gtest/gtest.h>

using namespace nncc::core::ADT;

namespace
{
class FeatureObjectTest : public ::testing::Test
{
protected:
  coco::FeatureObject *allocate()
  {
    auto o = new coco::FeatureObject{};
    _allocated.emplace_back(o);
    return o;
  }

  // TODO Deprecate this method
  coco::FeatureObject *allocate(const coco::FeatureShape &shape)
  {
    auto o = new coco::FeatureObject{};
    o->layout(coco::FeatureLayouts::Generic::create(shape));
    _allocated.emplace_back(o);
    return o;
  }

private:
  std::vector<std::unique_ptr<coco::FeatureObject>> _allocated;
};
} // namespace

TEST_F(FeatureObjectTest, ctor)
{
  const coco::FeatureShape shape{1, 3, 3};

  auto o = allocate(shape);

  ASSERT_EQ(o->shape(), shape);
  ASSERT_EQ(o->kind(), coco::Object::Kind::Feature);
}

// TODO Reimplement this test as a test for GenericFeatureLayout
#if 0
TEST_F(FeatureObjectTest, at)
{
  const uint32_t C = 1;
  const uint32_t H = 3;
  const uint32_t W = 3;

  const coco::FeatureShape shape{C, H, W};

  auto o = allocate(shape);

  coco::FeatureObject *mutable_ptr = o;
  const coco::FeatureObject *immutable_ptr = o;

  for (uint32_t ch = 0; ch < C; ++ch)
  {
    for (uint32_t row = 0; row < H; ++row)
    {
      for (uint32_t col = 0; col < W; ++col)
      {
        mutable_ptr->at(ch, row, col) = coco::ElemID{16};
      }
    }
  }

  for (uint32_t ch = 0; ch < C; ++ch)
  {
    for (uint32_t row = 0; row < H; ++row)
    {
      for (uint32_t col = 0; col < W; ++col)
      {
        ASSERT_EQ(immutable_ptr->at(ch, row, col).value(), 16);
      }
    }
  }
}
#endif

TEST_F(FeatureObjectTest, asFeature)
{
  const coco::FeatureShape shape{1, 3, 3};

  auto o = allocate(shape);

  coco::Object *mutable_object = o;
  const coco::Object *immutable_object = o;

  ASSERT_NE(mutable_object->asFeature(), nullptr);
  ASSERT_EQ(mutable_object->asFeature(), immutable_object->asFeature());
}

TEST_F(FeatureObjectTest, casting_helpers)
{
  auto obj = allocate();

  ASSERT_TRUE(coco::isa<coco::FeatureObject>(obj));
  ASSERT_EQ(coco::cast<coco::FeatureObject>(obj), obj);
  ASSERT_EQ(coco::safe_cast<coco::FeatureObject>(obj), obj);
}
