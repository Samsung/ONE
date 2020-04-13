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

#ifndef __MOCO_IR_TFSTOPGRADIENT_H__
#define __MOCO_IR_TFSTOPGRADIENT_H__

#include "moco/IR/TFNodeDecl.h"

namespace moco
{

/// @note TFStopGradient corresponds to the following GraphDef
/*
node {
  name: "StopGradient"
  op: "StopGradient"
  input: "Placeholder"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
*/

class TFStopGradient final : public FixedArityNode<1, TFNodeImpl<TFOpcode::StopGradient>>
{
public:
  TFStopGradient() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }
};

} // namespace moco

#endif // __MOCO_IR_TFSTOPGRADIENT_H__
