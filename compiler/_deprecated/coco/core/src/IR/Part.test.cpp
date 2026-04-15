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

#include "coco/IR/Part.h"
#include "coco/IR/Op.h"

#include <memory>

#include <gtest/gtest.h>

using std::make_unique;

namespace
{
namespace mock
{

// TODO Inherit UnaryOp instead of Op
struct Op final : public coco::Op
{
public:
  Op() : _arg{this}
  {
    // DO NOTHING
  }

public:
  uint32_t arity(void) const final { return 1; }
  coco::Op *arg(uint32_t n) const final { return arg(); }

  std::set<coco::Object *> uses() const override { throw std::runtime_error{"Not supported"}; }

public:
  ::coco::Op *arg(void) const { return _arg.child(); }
  void arg(::coco::Op *child) { _arg.child(child); }

private:
  coco::Part _arg;
};

} // namespace mock
} // namespace

TEST(PartTest, destructor)
{
  auto parent = make_unique<::mock::Op>();
  auto child = make_unique<::mock::Op>();

  parent->arg(child.get());
  ASSERT_EQ(parent->arg(), child.get());
  ASSERT_EQ(child->up(), parent.get());

  parent.reset();

  // NOTE parent SHOULD unlink itself from child on destruction
  ASSERT_EQ(child->up(), nullptr);
}
