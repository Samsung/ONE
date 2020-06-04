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

#ifndef __LOCOEX_IR_TFLNODE_VISITOR_H__
#define __LOCOEX_IR_TFLNODE_VISITOR_H__

#include "TFLNode.h"

#include <oops/InternalExn.h>

namespace locoex
{

/**
 * DO NOT use this class. Use TFLNodeVisitor instead.
 */
template <typename T> struct TFLNodeVisitorBase
{
  virtual ~TFLNodeVisitorBase() = default;

#define TFL_NODE(OPCODE, TFL_CLASS) virtual T visit(const TFL_CLASS *) = 0;

#include "TFLNodes.lst"
#undef TFL_NODE
};

template <typename T> struct TFLNodeVisitor : public TFLNodeVisitorBase<T>
{
  virtual ~TFLNodeVisitor() = default;

#define TFL_NODE(OPCODE, TFL_CLASS) \
                                    \
  virtual T visit(const TFL_CLASS *node) { return visit(static_cast<const TFLNode *>(node)); }

#include "TFLNodes.lst"
#undef TFL_NODE

  /// @brief Default fallback
  virtual T visit(const TFLNode *) { INTERNAL_EXN("TFLNodeVisitor: NYI node"); }
};

/**
 * DO NOT use this class. Use TFLNodeMutableVisitor instead.
 */
template <typename T> struct TFLNodeMutableVisitorBase
{
  virtual ~TFLNodeMutableVisitorBase() = default;

#define TFL_NODE(OPCODE, TFL_CLASS) virtual T visit(TFL_CLASS *) = 0;

#include "TFLNodes.lst"
#undef TFL_NODE
};

template <typename T> struct TFLNodeMutableVisitor : public TFLNodeMutableVisitorBase<T>
{
  virtual ~TFLNodeMutableVisitor() = default;

#define TFL_NODE(OPCODE, TFL_CLASS) \
                                    \
  virtual T visit(TFL_CLASS *node) { return visit(static_cast<TFLNode *>(node)); }

#include "TFLNodes.lst"
#undef TFL_NODE

  /// @brief Default fallback
  virtual T visit(TFLNode *) { INTERNAL_EXN("TFLNodeMutableVisitor: NYI node"); }
};

} // namespace locoex

#endif // __LOCOEX_IR_TFLNODE_VISITOR_H__
