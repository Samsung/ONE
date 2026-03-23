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

#ifndef __MOCO_IR_TFBIASADD_H__
#define __MOCO_IR_TFBIASADD_H__

#include "moco/IR/TFNodeDecl.h"

namespace moco
{

/// @note TFBiasAdd corresponds to the following GraphDef
/*
node {
  name: "bias_add_01"
  op: "BiasAdd"
  input: "input_01"
  input: "bias_add_01/bias"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "data_format"
    value {
      s: "NHWC"
    }
  }
}
*/

class TFBiasAdd final : public FixedArityNode<2, TFNodeImpl<TFOpcode::BiasAdd>>
{
public:
  TFBiasAdd() = default;

public:
  Node *value(void) const { return at(0)->node(); }
  void value(Node *node) { return at(0)->node(node); }

  Node *bias(void) const { return at(1)->node(); }
  void bias(Node *node) { return at(1)->node(node); }

  const TFDataLayout data_layout(void) const { return _data_layout; }
  void data_layout(const TFDataLayout &data_layout) { _data_layout = data_layout; }

private:
  TFDataLayout _data_layout;
};

} // namespace moco

#endif // __MOCO_IR_TFBIASADD_H__
