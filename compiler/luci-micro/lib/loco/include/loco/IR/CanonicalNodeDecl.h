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

#ifndef __LOCO_IR_CANONICAL_NODE_DECL_H__
#define __LOCO_IR_CANONICAL_NODE_DECL_H__

#include "loco/IR/Node.h"
#include "loco/IR/Dialect.h"
#include "loco/IR/CanonicalOpcode.h"
#include "loco/IR/CanonicalNodeVisitor.forward.h"

namespace loco
{

struct CanonicalNode : public Node
{
  virtual ~CanonicalNode() = default;

  const Dialect *dialect(void) const final;
  virtual CanonicalOpcode opcode(void) const = 0;

  template <typename T> T accept(CanonicalNodeVisitorBase<T> *) const;
  template <typename T> T accept(CanonicalNodeMutableVisitorBase<T> *);
};

template <CanonicalOpcode Code, template <typename T> class... Mixins>
struct CanonicalNodeDef : public virtual CanonicalNode, public Mixins<CanonicalNode>...
{
  virtual ~CanonicalNodeDef() = default;

  uint32_t opnum(void) const final { return static_cast<uint32_t>(Code); }
  CanonicalOpcode opcode(void) const final { return Code; }
};

} // namespace loco

#endif // __LOCO_IR_CANONICAL_NODE_H__
