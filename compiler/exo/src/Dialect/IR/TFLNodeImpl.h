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

#ifndef __LOCOEX_IR_TFLNODEIMPL_H__
#define __LOCOEX_IR_TFLNODEIMPL_H__

#include "TFLNodes.h"

#include <oops/InternalExn.h>

#include <cassert>

namespace locoex
{

template <typename T> T TFLNode::accept(TFLNodeVisitorBase<T> *v) const
{
  switch (this->opcode())
  {
#define TFL_NODE(OPCODE, CLASS) \
                                \
  case TFLOpcode::OPCODE:       \
    return v->visit(dynamic_cast<const CLASS *>(this));

#include "TFLNodes.lst"
#undef TFL_NODE

    default:
      break;
  }

  INTERNAL_EXN("TFLNode::accept(TFLNodeVisitorBase) not handled");
}

template <typename T> T TFLNode::accept(TFLNodeMutableVisitorBase<T> *v)
{
  switch (this->opcode())
  {
#define TFL_NODE(OPCODE, CLASS) \
                                \
  case TFLOpcode::OPCODE:       \
    return v->visit(dynamic_cast<CLASS *>(this));

#include "TFLNodes.lst"
#undef TFL_NODE

    default:
      break;
  }

  INTERNAL_EXN("TFLNode::accept(TFLNodeMutableVisitorBase) not handled");
}

} // namespace locoex

#endif // __LOCOEX_IR_TFLNODEIMPL_H__
