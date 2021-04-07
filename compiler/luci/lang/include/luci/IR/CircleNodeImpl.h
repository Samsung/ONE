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

#ifndef __LUCI_IR_CIRCLENODEIMPL_H__
#define __LUCI_IR_CIRCLENODEIMPL_H__

#include "CircleNodes.h"

#include <oops/InternalExn.h>

#include <cassert>

namespace luci
{

template <typename T> T CircleNode::accept(CircleNodeVisitorBase<T> *v) const
{
  switch (this->opcode())
  {
#define CIRCLE_NODE(OPCODE, CLASS) \
                                   \
  case CircleOpcode::OPCODE:       \
    return v->visit(dynamic_cast<const CLASS *>(this));
#define CIRCLE_VNODE CIRCLE_NODE

#include "CircleNodes.lst"
#undef CIRCLE_VNODE
#undef CIRCLE_NODE

    default:
      break;
  }

  INTERNAL_EXN("CircleNode::accept(CircleNodeVisitorBase) not handled");
}

template <typename T> T CircleNode::accept(CircleNodeMutableVisitorBase<T> *v)
{
  switch (this->opcode())
  {
#define CIRCLE_NODE(OPCODE, CLASS) \
                                   \
  case CircleOpcode::OPCODE:       \
    return v->visit(dynamic_cast<CLASS *>(this));
#define CIRCLE_VNODE CIRCLE_NODE

#include "CircleNodes.lst"
#undef CIRCLE_VNODE
#undef CIRCLE_NODE

    default:
      break;
  }

  INTERNAL_EXN("CircleNode::accept(CircleNodeMutableVisitorBase) not handled");
}

} // namespace luci

#endif // __LUCI_IR_CIRCLENODEIMPL_H__
