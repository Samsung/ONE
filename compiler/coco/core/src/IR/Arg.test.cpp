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

#include "coco/IR/Arg.h"

#include <nncc/core/ADT/tensor/IndexEnumerator.h>
#include <nncc/core/ADT/tensor/LexicalLayout.h>

#include <memory>
#include <vector>

#include <gtest/gtest.h>

using nncc::core::ADT::tensor::Index;
using nncc::core::ADT::tensor::IndexEnumerator;
using nncc::core::ADT::tensor::LexicalLayout;
using nncc::core::ADT::tensor::Shape;

namespace
{
class ArgTest : public ::testing::Test
{
protected:
  coco::Arg *allocate(const Shape &shape)
  {
    auto arg = new coco::Arg{shape};
    _allocated.emplace_back(arg);
    return arg;
  }

private:
  std::vector<std::unique_ptr<coco::Arg>> _allocated;
};
} // namespace

TEST_F(ArgTest, constructor)
{
  const Shape shape{1, 3, 3, 1};

  auto arg = allocate(shape);

  ASSERT_EQ(arg->shape(), shape);
  ASSERT_TRUE(arg->name().empty());
  ASSERT_EQ(arg->bag(), nullptr);
}

TEST_F(ArgTest, name_update)
{
  const Shape shape{1, 3, 3, 1};

  auto arg = allocate(shape);

  arg->name("data");
  ASSERT_EQ(arg->name(), "data");
}

TEST_F(ArgTest, at)
{
  const Shape shape{1, 3, 3, 1};

  auto arg = allocate(shape);

  coco::Arg *mutable_ptr = arg;
  const coco::Arg *immutable_ptr = arg;

  for (IndexEnumerator e{shape}; e.valid(); e.advance())
  {
    mutable_ptr->at(e.current()) = coco::ElemID{16};
  }

  for (IndexEnumerator e{shape}; e.valid(); e.advance())
  {
    ASSERT_EQ(immutable_ptr->at(e.current()).value(), 16);
  }
}

TEST_F(ArgTest, reorder)
{
  const Shape shape{2, 2, 2, 2};

  auto arg = allocate(shape);

  arg->reorder<LexicalLayout>();

  ASSERT_EQ(arg->at(Index{0, 0, 0, 0}).value(), 0);
  ASSERT_EQ(arg->at(Index{0, 0, 0, 1}).value(), 1);
}
