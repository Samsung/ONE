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

#ifndef __MOCO_IR_TFPLACEHOLDER_H__
#define __MOCO_IR_TFPLACEHOLDER_H__

#include "moco/IR/TFNodeDecl.h"

#include <loco/IR/DataTypeTraits.h>
#include <loco/IR/NodeMixins.h>
#include <loco/IR/GraphInputIndex.h>
#include <loco/IR/TensorShape.h>

namespace moco
{

/// @note TFPlaceholder corresponds to the following GraphDef
/*
node {
  name: "placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 3
        }
        dim {
          size: 3
        }
        dim {
          size: 1
        }
      }
    }
  }
}
*/

/**
 * @brief  IR for tf.placeholder
 */
class TFPlaceholder final : public FixedArityNode<0, TFNodeImpl<TFOpcode::Placeholder>>,
                            public loco::NodeMixin<loco::NodeTrait::DataType>,
                            public loco::NodeMixin<loco::NodeTrait::TensorShape>
{
public:
  TFPlaceholder() = default;

  // TODO Update unkown shape information. tensorflow::NodeDef may not have "shape" attr.
};

} // namespace moco

namespace moco
{

bool indexed(const TFPlaceholder *node);
loco::GraphInputIndex index(const TFPlaceholder *node);
void index(TFPlaceholder *node, const loco::GraphInputIndex index);
loco::TensorShape tensor_shape(const TFPlaceholder *node);

TFPlaceholder *placeholder_node(loco::Graph *g, const loco::GraphInputIndex &idx);

} // namespace moco

#endif // __MOCO_IR_TFPLACEHOLDER_H__
