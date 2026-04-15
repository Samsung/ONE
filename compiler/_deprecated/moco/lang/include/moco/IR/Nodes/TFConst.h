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

#ifndef __MOCO_IR_TFCONSTANT_H__
#define __MOCO_IR_TFCONSTANT_H__

#include "moco/IR/TFNodeDecl.h"

#include <loco/IR/DataTypeTraits.h>
#include <loco/IR/NodeMixins.h>
#include <loco/IR/TensorShape.h>

#include <vector>

namespace moco
{

/// @note TFConst corresponds to the following GraphDef
/*
node {
  name: "val"
  op: "Const"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim { size: 1 }
          dim { size: 3 }
          dim { size: 4 }
          dim { size: 4 }
        }
        float_val: 2.1
      }
    }
  }
}
*/

/**
 * @brief  IR for tf.constant
 *
 * @note   Implementation for this class came from Canonical ConstGen
 *         Read comments in loco::ConstGen for details
 */
class TFConst final : public FixedArityNode<0, TFNodeImpl<TFOpcode::Const>>,
                      public loco::NodeMixin<loco::NodeTrait::DataType>,
                      public loco::NodeMixin<loco::NodeTrait::TensorShape>
{
public:
  TFConst() = default;

public:
  template <loco::DataType DT> uint32_t size(void) const;
  template <loco::DataType DT> void size(uint32_t size);

  template <loco::DataType DT> const typename loco::DataTypeImpl<DT>::Type &at(uint32_t n) const;
  template <loco::DataType DT> typename loco::DataTypeImpl<DT>::Type &at(uint32_t n);

private:
  std::vector<uint8_t> _data;
};

} // namespace moco

namespace moco
{

loco::TensorShape tensor_shape(const TFConst *node);

uint32_t num_elements(const TFConst *tfconst);
bool same_shape(const TFConst *lhs, const TFConst *rhs);

} // namespace moco

#endif // __MOCO_IR_TFCONSTANT_H__
