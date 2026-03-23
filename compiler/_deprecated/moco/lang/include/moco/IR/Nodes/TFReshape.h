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

#ifndef __MOCO_IR_TFRESHAPE_H__
#define __MOCO_IR_TFRESHAPE_H__

#include "moco/IR/TFNodeDecl.h"

namespace moco
{

/// @note TFReshape corresponds to the following GraphDef
/*
node {
  name: "reshape"
  op: "Reshape"
  input: "tensor"
  input: "shape"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
}
*/

class TFReshape final : public FixedArityNode<2, TFNodeImpl<TFOpcode::Reshape>>
{
public:
  TFReshape() = default;

public:
  Node *tensor(void) const { return at(0)->node(); }
  void tensor(Node *node) { at(0)->node(node); }

  Node *shape(void) const { return at(1)->node(); }
  void shape(Node *node) { at(1)->node(node); }
};

} // namespace moco

#endif // __MOCO_IR_TFRESHAPE_H__
