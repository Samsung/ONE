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

#ifndef __MOCO_IR_TFNODE_VISITOR_H__
#define __MOCO_IR_TFNODE_VISITOR_H__

#include "moco/IR/TFNodes.h"

#include <stdexcept>

namespace moco
{

/**
 * DO NOT use this class. Use TFNodeVisitor instead.
 */
template <typename T> struct TFNodeVisitorBase
{
  virtual ~TFNodeVisitorBase() = default;

#define TENSORFLOW_NODE(OPCODE, CLASS) virtual T visit(const CLASS *) = 0;
#include "TFNodes.lst"
#undef TENSORFLOW_NODE
};

template <typename T> struct TFNodeVisitor : public TFNodeVisitorBase<T>
{
  virtual ~TFNodeVisitor() = default;

#define TENSORFLOW_NODE(OPCODE, CLASS) \
  virtual T visit(const CLASS *node) { return visit(static_cast<const TFNode *>(node)); }
#include "TFNodes.lst"
#undef TENSORFLOW_NODE

  // TODO including oops will make oops dependent to modules that include this
  // postpone decision to this or not
  /// @brief Default fallback
  virtual T visit(const TFNode *) { throw std::runtime_error{"Unsupported Node"}; }
};

/**
 * DO NOT use this class. Use TFNodeMutableVisitor instead.
 */
template <typename T> struct TFNodeMutableVisitorBase
{
  virtual ~TFNodeMutableVisitorBase() = default;

#define TENSORFLOW_NODE(OPCODE, CLASS) virtual T visit(CLASS *) = 0;
#include "TFNodes.lst"
#undef TENSORFLOW_NODE
};

template <typename T> struct TFNodeMutableVisitor : public TFNodeMutableVisitorBase<T>
{
  virtual ~TFNodeMutableVisitor() = default;

#define TENSORFLOW_NODE(OPCODE, CLASS) \
  virtual T visit(CLASS *node) { return visit(static_cast<TFNode *>(node)); }
#include "TFNodes.lst"
#undef TENSORFLOW_NODE

  // TODO including oops will make oops dependent to modules that include this
  // postpone decision to this or not
  /// @brief Default fallback
  virtual T visit(TFNode *) { throw std::runtime_error{"Unsupported Node"}; }
};

} // namespace moco

#endif // __MOCO_IR_TFNODE_VISITOR_H__
