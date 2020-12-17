/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _MIR_OPS_CONCAT_OP_H_
#define _MIR_OPS_CONCAT_OP_H_

#include "mir/Operation.h"

namespace mir
{
namespace ops
{

/**
 * @brief Description of tensor concatenation operation.
 */
class ConcatOp : public Operation
{
public:
  ConcatOp(const std::vector<Output *> &args, int32_t axis)
    : Operation(Type::concat, args), _axis(axis)
  {
    inferOutputTypes();
  }

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    return new ConcatOp(inputs, _axis);
  };

  int32_t getAxis() const
  {
    if (_axis < 0)
    {
      // Negative axis is used to index starting from the last element of the shape
      // -1 means last element, -2 means second from end, like in python
      int32_t res = _axis + getInputShape(0).rank();
      assert(res >= 0);
      return res;
    }
    return _axis;
  }

private:
  void inferOutputTypes();

  /// @brief The axis along which to concatenate, may be negative to index from the end
  int32_t _axis;
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_CONCAT_OP_H_
