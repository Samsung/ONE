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

#ifndef __LOCOEX_IR_CIRCLENODE_VISITOR_H__
#define __LOCOEX_IR_CIRCLENODE_VISITOR_H__

#include "CircleNode.h"
#include "CircleNodes.h"

#include <oops/InternalExn.h>

namespace locoex
{

/**
 * DO NOT use this class. Use CircleNodeVisitor instead.
 */
template <typename T> struct CircleNodeVisitorBase
{
  virtual ~CircleNodeVisitorBase() = default;

#define CIRCLE_NODE(OPCODE, Circle_CLASS) virtual T visit(const Circle_CLASS *) = 0;

#include "CircleNodes.lst"
#undef CIRCLE_NODE
};

template <typename T> struct CircleNodeVisitor : public CircleNodeVisitorBase<T>
{
  virtual ~CircleNodeVisitor() = default;

#define CIRCLE_NODE(OPCODE, Circle_CLASS) \
                                          \
  virtual T visit(const Circle_CLASS *node) { return visit(static_cast<const CircleNode *>(node)); }

#include "CircleNodes.lst"
#undef CIRCLE_NODE

  /// @brief Default fallback
  virtual T visit(const CircleNode *) { INTERNAL_EXN("CircleNodeVisistor: NYI node"); }
};

/**
 * DO NOT use this class. Use CircleNodeMutableVisitor instead.
 */
template <typename T> struct CircleNodeMutableVisitorBase
{
  virtual ~CircleNodeMutableVisitorBase() = default;

#define CIRCLE_NODE(OPCODE, Circle_CLASS) virtual T visit(Circle_CLASS *) = 0;

#include "CircleNodes.lst"
#undef CIRCLE_NODE
};

template <typename T> struct CircleNodeMutableVisitor : public CircleNodeMutableVisitorBase<T>
{
  virtual ~CircleNodeMutableVisitor() = default;

#define CIRCLE_NODE(OPCODE, Circle_CLASS) \
                                          \
  virtual T visit(Circle_CLASS *node) { return visit(static_cast<CircleNode *>(node)); }

#include "CircleNodes.lst"
#undef CIRCLE_NODE

  /// @brief Default fallback
  virtual T visit(CircleNode *) { INTERNAL_EXN("CircleMutableNodeVisistor: NYI node"); }
};

} // namespace locoex

#endif // __LOCOEX_IR_CIRCLENODE_VISITOR_H__
