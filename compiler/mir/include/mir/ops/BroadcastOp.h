/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef _MIR_OPS_BROADCAST_OP_H_
#define _MIR_OPS_BROADCAST_OP_H_

#include "mir/Operation.h"

namespace mir
{
namespace ops
{

class BroadcastOp : public Operation
{
public:
  BroadcastOp(Output *input, const Shape &target_shape) : Operation(Type::broadcast, {input})
  {
    inferOutputTypes(target_shape);
  }

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    return new BroadcastOp(inputs[0], getOutputShape(0));
  }

private:
  void inferOutputTypes(const Shape &target_shape);
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_BINARY_BROADCAST_OP_H_
