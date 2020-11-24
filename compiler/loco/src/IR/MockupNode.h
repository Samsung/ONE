/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOCO_IR_MOCKUP_NODE_H__
#define __LOCO_IR_MOCKUP_NODE_H__

#include "loco/IR/Use.h"
#include "loco/IR/Node.h"

namespace
{

struct MockDialect final : public loco::Dialect
{
  static loco::Dialect *get(void)
  {
    static MockDialect d;
    return &d;
  }
};

/// @brief Mockup node for internal testing
class MockupNode final : public loco::Node
{
public:
  MockupNode() = default;

public:
  const loco::Dialect *dialect(void) const final { return MockDialect::get(); }
  uint32_t opnum(void) const final { return 0; }

  uint32_t arity(void) const final { return 1; }
  Node *arg(uint32_t N) const final { return _arg.node(); }
  void drop(void) final { _arg.node(nullptr); }

  Node *in(void) const { return _arg.node(); }
  void in(Node *node) { _arg.node(node); }

private:
  loco::Use _arg{this};
};

/// @brief Mockup2Node node for internal testing
class Mockup2Node final : public loco::Node
{
public:
  Mockup2Node() = default;

public:
  const loco::Dialect *dialect(void) const final { return MockDialect::get(); }
  uint32_t opnum(void) const final { return 1; }

  uint32_t arity(void) const final { return 0; }
  Node *arg(uint32_t) const final { return nullptr; }
  void drop(void) final {}
};

} // namespace

#endif // __LOCO_IR_MOCKUP_NODE_H__
