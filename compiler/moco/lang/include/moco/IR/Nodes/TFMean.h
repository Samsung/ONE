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

#ifndef __MOCO_IR_TFMEAN_H__
#define __MOCO_IR_TFMEAN_H__

#include "moco/IR/TFNodeDecl.h"

#include <vector>

namespace moco
{

/// @note TFMean corresponds to the following GraphDef
/*
node {
  name: "Mean"
  op: "Mean"
  input: "Placeholder"
  input: "Mean/reduction_indices"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
  attr {
    key: "Tidx"
    value { type: DT_INT32 }
  }
  attr {
    key: "keep_dims"
    value { b: true }
  }
}
*/

class TFMean final : public FixedArityNode<2, TFNodeImpl<TFOpcode::Mean>>
{
public:
  TFMean() = default;

public:
  Node *input(void) const { return at(0)->node(); }
  void input(Node *node) { at(0)->node(node); }

  Node *reduction_indices(void) const { return at(1)->node(); }
  void reduction_indices(Node *node) { at(1)->node(node); }

public:
  bool keep_dims(void) const { return _keep_dims; }
  void keep_dims(bool keep_dims) { _keep_dims = keep_dims; }

private:
  bool _keep_dims = false;
};

} // namespace moco

#endif // __MOCO_IR_TFMEAN_H__
