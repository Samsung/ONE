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

#include "coco/IR/Output.h"
#include "coco/IR/BagManager.h"

#include <nncc/core/ADT/tensor/IndexEnumerator.h>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::Shape;

TEST(IR_OUTPUT, ctor_should_set_shape)
{
  const nncc::core::ADT::tensor::Shape shape{1, 3, 3, 1};
  coco::Output output{shape};

  ASSERT_EQ(output.shape(), shape);
}

TEST(IR_OUTPUT, bag_update)
{
  // Create a bag for test
  coco::BagManager bag_mgr;

  auto bag = bag_mgr.create(9);

  const nncc::core::ADT::tensor::Shape shape{1, 3, 3, 1};
  coco::Output output{shape};

  output.bag(bag);
  ASSERT_EQ(output.bag(), bag);

  // bag(...) method SHOULD update 'bag' type
  ASSERT_TRUE(bag->isOutput());

  output.bag(nullptr);

  // bag(nullptr) SHOULD revert 'bag' type
  ASSERT_FALSE(bag->isOutput());
}

TEST(IR_OUTPUT, name_update)
{
  const nncc::core::ADT::tensor::Shape shape{1, 3, 3, 1};
  coco::Output output{shape};

  output.name("softmax");
  ASSERT_EQ(output.name(), "softmax");
}

TEST(IR_OUTPUT, at)
{
  const Shape shape{1, 3, 3, 1};
  coco::Output input{shape};

  coco::Output *mutable_ptr = &input;
  const coco::Output *immutable_ptr = &input;

  for (IndexEnumerator e{shape}; e.valid(); e.advance())
  {
    mutable_ptr->at(e.current()) = coco::ElemID{16};
  }

  for (IndexEnumerator e{shape}; e.valid(); e.advance())
  {
    ASSERT_EQ(immutable_ptr->at(e.current()).value(), 16);
  }
}
