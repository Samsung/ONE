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

#ifndef __MOCO_IR_TFSTRIDEDSLICE_H__
#define __MOCO_IR_TFSTRIDEDSLICE_H__

#include "moco/IR/TFNodeDecl.h"

namespace moco
{

/// @note TFStridedSlice corresponds to the following GraphDef
/*
node {
  name: "StridedSlice"
  op: "StridedSlice"
  input: "input"
  input: "begin"
  input: "end"
  input: "stride"
  attr {
    key: "Index"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "begin_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "ellipsis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "end_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "new_axis_mask"
    value {
      i: 0
    }
  }
  attr {
    key: "shrink_axis_mask"
    value {
      i: 0
    }
  }
}
*/

class TFStridedSlice final : public FixedArityNode<4, TFNodeImpl<TFOpcode::StridedSlice>>
{
public:
  TFStridedSlice() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

  Node *begin(void) const { return at(1)->node(); }
  void begin(Node *node) { at(1)->node(node); }

  Node *end(void) const { return at(2)->node(); }
  void end(Node *node) { at(2)->node(node); }

  Node *strides(void) const { return at(3)->node(); }
  void strides(Node *node) { at(3)->node(node); }

public:
  int32_t begin_mask(void) const { return _begin_mask; }
  void begin_mask(int32_t begin_mask) { _begin_mask = begin_mask; }

  int32_t end_mask(void) const { return _end_mask; }
  void end_mask(int32_t end_mask) { _end_mask = end_mask; }

  int32_t ellipsis_mask(void) const { return _ellipsis_mask; }
  void ellipsis_mask(int32_t ellipsis_mask) { _ellipsis_mask = ellipsis_mask; }

  int32_t new_axis_mask(void) const { return _new_axis_mask; }
  void new_axis_mask(int32_t new_axis_mask) { _new_axis_mask = new_axis_mask; }

  int32_t shrink_axis_mask(void) const { return _shrink_axis_mask; }
  void shrink_axis_mask(int32_t shrink_axis_mask) { _shrink_axis_mask = shrink_axis_mask; }

private:
  int32_t _begin_mask{0};
  int32_t _end_mask{0};
  int32_t _ellipsis_mask{0};
  int32_t _new_axis_mask{0};
  int32_t _shrink_axis_mask{0};
};

} // namespace moco

#endif // __MOCO_IR_TFSTRIDEDSLICE_H__
