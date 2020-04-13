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

#ifndef __LOCO_IR_CANONICAL_NODE_VISITOR_H__
#define __LOCO_IR_CANONICAL_NODE_VISITOR_H__

#include "loco/IR/Nodes.h"

#include <stdexcept>

namespace loco
{

/**
 * DO NOT use this class. Use CanonicalNodeVisitor instead.
 */
template <typename T> struct CanonicalNodeVisitorBase
{
  virtual ~CanonicalNodeVisitorBase() = default;

#define CANONICAL_NODE(OPCODE, CLASS) virtual T visit(const CLASS *) = 0;
#include "CanonicalNodes.lst"
#undef CANONICAL_NODE
};

template <typename T> struct CanonicalNodeVisitor : public CanonicalNodeVisitorBase<T>
{
  virtual ~CanonicalNodeVisitor() = default;

#define CANONICAL_NODE(OPCODE, CLASS) \
  virtual T visit(const CLASS *node) { return visit(static_cast<const Node *>(node)); }
#include "CanonicalNodes.lst"
#undef CANONICAL_NODE

  /// @brief Default fallback
  virtual T visit(const Node *) { throw std::runtime_error{"Not implemented, yet"}; }
};

/**
 * DO NOT use this class. Use CanonicalNodeMutableVisitor instead.
 */
template <typename T> struct CanonicalNodeMutableVisitorBase
{
  virtual ~CanonicalNodeMutableVisitorBase() = default;

#define CANONICAL_NODE(OPCODE, CLASS) virtual T visit(CLASS *) = 0;
#include "CanonicalNodes.lst"
#undef CANONICAL_NODE
};

template <typename T> struct CanonicalNodeMutableVisitor : public CanonicalNodeMutableVisitorBase<T>
{
  virtual ~CanonicalNodeMutableVisitor() = default;

#define CANONICAL_NODE(OPCODE, CLASS) \
  virtual T visit(CLASS *node) { return visit(static_cast<Node *>(node)); }
#include "CanonicalNodes.lst"
#undef CANONICAL_NODE

  /// @brief Default fallback
  virtual T visit(Node *) { throw std::runtime_error{"Not implemented, yet"}; }
};

} // namespace loco

#endif // __LOCO_IR_CANONICAL_NODE_VISITOR_H__
