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

#ifndef _MIR_OPS_CONSTANT_OP_H_
#define _MIR_OPS_CONSTANT_OP_H_

#include "mir/Operation.h"
#include "mir/TensorVariant.h"

namespace mir
{
namespace ops
{

class ConstantOp : public Operation
{
public:
  explicit ConstantOp(const TensorVariant &value) : Operation(Type::constant, {}), _value(value)
  {
    setOutputType(0, _value.getType());
  }

  const TensorVariant &getValue() const { return _value; }

  Operation *copyWithInputs(const std::vector<mir::Operation::Output *> &input) override
  {
    assert(false && "Copying constants is not allowed!");
    (void)input;
    return nullptr;
  }

private:
  TensorVariant _value;
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_CONSTANT_OP_H_
