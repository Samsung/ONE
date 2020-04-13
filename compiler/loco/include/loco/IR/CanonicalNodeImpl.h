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

#ifndef __LOCO_IR_CANONICAL_NODE_IMPL_H__
#define __LOCO_IR_CANONICAL_NODE_IMPL_H__

#include "loco/IR/Nodes.h"
#include "loco/IR/CanonicalNodeVisitor.h"

#include <stdexcept>

namespace loco
{

template <typename T> T CanonicalNode::accept(CanonicalNodeVisitorBase<T> *v) const
{
  switch (this->opcode())
  {
#define CANONICAL_NODE(OPCODE, CLASS) \
  case CanonicalOpcode::OPCODE:       \
    return v->visit(dynamic_cast<const CLASS *>(this));

#include "CanonicalNodes.lst"
#undef CANONICAL_NODE
    default:
      break;
  }

  throw std::runtime_error{"NYI"};
}

template <typename T> T CanonicalNode::accept(CanonicalNodeMutableVisitorBase<T> *v)
{
  switch (this->opcode())
  {
#define CANONICAL_NODE(OPCODE, CLASS) \
  case CanonicalOpcode::OPCODE:       \
    return v->visit(dynamic_cast<CLASS *>(this));

#include "CanonicalNodes.lst"
#undef CANONICAL_NODE
    default:
      break;
  }

  throw std::runtime_error{"NYI"};
}

} // namespace loco

#endif // __LOCO_IR_CANONICAL_NODE_IMPL_H__
