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

#include "coco/IR/ObjectManager.h"
#include "coco/IR/BagManager.h"

#include "coco/IR/FeatureObject.h"
#include "coco/IR/KernelObject.h"

#include <gtest/gtest.h>

TEST(IR_OBJECT_MANAGER, create_feature_with_template)
{
  coco::ObjectManager mgr;

  auto feature = mgr.create<coco::FeatureObject>();

  ASSERT_EQ(feature->layout(), nullptr);
}

TEST(IR_OBJECT_MANAGER, create_kernel_with_template)
{
  coco::ObjectManager mgr;

  auto kernel = mgr.create<coco::KernelObject>();

  ASSERT_EQ(kernel->layout(), nullptr);
}

TEST(IR_OBJECT_MANAGER, destroy)
{
  coco::BagManager bag_mgr;
  coco::ObjectManager obj_mgr;

  auto bag = bag_mgr.create(3);
  auto feature = obj_mgr.create<coco::FeatureObject>();

  feature->bag(bag);

  obj_mgr.destroy(feature);

  // Object SHOULD BE unlinked from its dependent bag on destruction
  ASSERT_EQ(bag->deps()->size(), 0);
}
