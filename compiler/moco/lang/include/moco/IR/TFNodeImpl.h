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

#ifndef __MOCO_IR_TFNODE_IMPL_H__
#define __MOCO_IR_TFNODE_IMPL_H__

#include "moco/IR/TFNodes.h"
#include "moco/IR/TFNodeVisitor.h"

#include <stdexcept>

namespace moco
{

template <typename T> T TFNode::accept(TFNodeVisitorBase<T> *v) const
{
  switch (this->opcode())
  {
#define TENSORFLOW_NODE(OPCODE, CLASS) \
  case TFOpcode::OPCODE:               \
    return v->visit(dynamic_cast<const CLASS *>(this));

#include "TFNodes.lst"
#undef TENSORFLOW_NODE
    default:
      break;
  }

  // TODO including oops will make oops dependent to modules that include this
  // postpone decision to this or not
  throw std::runtime_error{"Unsupported Node"};
}

template <typename T> T TFNode::accept(TFNodeMutableVisitorBase<T> *v)
{
  switch (this->opcode())
  {
#define TENSORFLOW_NODE(OPCODE, CLASS) \
  case TFOpcode::OPCODE:               \
    return v->visit(dynamic_cast<CLASS *>(this));

#include "TFNodes.lst"
#undef TENSORFLOW_NODE
    default:
      break;
  }

  // TODO including oops will make oops dependent to modules that include this
  // postpone decision to this or not
  throw std::runtime_error{"Unsupported Node"};
}

} // namespace moco

#endif // __MOCO_IR_TFNODE_IMPL_H__
