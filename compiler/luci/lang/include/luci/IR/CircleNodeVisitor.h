/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LUCI_IR_CIRCLENODE_VISITOR_H__
#define __LUCI_IR_CIRCLENODE_VISITOR_H__

#include "CircleNode.h"
#include "CircleNodes.h"

#include <oops/InternalExn.h>

namespace luci
{

/**
 * DO NOT use this class. Use CircleNodeVisitor instead.
 */
template <typename T> struct CircleNodeVisitorBase
{
  virtual ~CircleNodeVisitorBase() = default;

#define CIRCLE_NODE(OPCODE, CIRCLE_CLASS) virtual T visit(const CIRCLE_CLASS *) = 0;
#define CIRCLE_VNODE CIRCLE_NODE

#include "CircleNodes.lst"
#undef CIRCLE_VNODE
#undef CIRCLE_NODE
};

template <typename T> struct CircleNodeVisitor : public CircleNodeVisitorBase<T>
{
  virtual ~CircleNodeVisitor() = default;

#define CIRCLE_NODE(OPCODE, CIRCLE_CLASS) \
  virtual T visit(const CIRCLE_CLASS *node) { return visit(static_cast<const CircleNode *>(node)); }
#define CIRCLE_VNODE CIRCLE_NODE

#include "CircleNodes.lst"

#undef CIRCLE_VNODE
#undef CIRCLE_NODE

  /// @brief Default fallback
  virtual T visit(const CircleNode *) { INTERNAL_EXN("CircleNodeVisitor: NYI node"); }
};

/**
 * DO NOT use this class. Use CircleNodeMutableVisitor instead.
 */
template <typename T> struct CircleNodeMutableVisitorBase
{
  virtual ~CircleNodeMutableVisitorBase() = default;

#define CIRCLE_NODE(OPCODE, CIRCLE_CLASS) virtual T visit(CIRCLE_CLASS *) = 0;
#define CIRCLE_VNODE CIRCLE_NODE

#include "CircleNodes.lst"

#undef CIRCLE_VNODE
#undef CIRCLE_NODE
};

template <typename T> struct CircleNodeMutableVisitor : public CircleNodeMutableVisitorBase<T>
{
  virtual ~CircleNodeMutableVisitor() = default;

#define CIRCLE_NODE(OPCODE, CIRCLE_CLASS) \
  virtual T visit(CIRCLE_CLASS *node) { return visit(static_cast<CircleNode *>(node)); }
#define CIRCLE_VNODE CIRCLE_NODE

#include "CircleNodes.lst"

#undef CIRCLE_VNODE
#undef CIRCLE_NODE

  /// @brief Default fallback
  virtual T visit(CircleNode *) { INTERNAL_EXN("CircleNodeMutableVisitor: NYI node"); }
};

} // namespace luci

#endif // __LUCI_IR_CircleNode_VISITOR_H__
