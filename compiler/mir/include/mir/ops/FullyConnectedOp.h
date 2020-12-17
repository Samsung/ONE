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

#ifndef _MIR_OPS_FULLY_CONNECTED_OP_H_
#define _MIR_OPS_FULLY_CONNECTED_OP_H_

#include "mir/Operation.h"
#include "mir/TensorVariant.h"

namespace mir
{
namespace ops
{

class FullyConnectedOp : public Operation
{
public:
  FullyConnectedOp(Output *input, Output *weights)
    : Operation(Type::fullyConnected, {input, weights})
  {
    inferOutputTypes();
  }

  FullyConnectedOp(Output *input, Output *weights, Output *bias)
    : Operation(Type::fullyConnected, {input, weights, bias})
  {
    inferOutputTypes();
  }

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    if (inputs.size() == 2)
      return new FullyConnectedOp(inputs[0], inputs[1]);
    else
      return new FullyConnectedOp(inputs[0], inputs[1], inputs[2]);
  }

private:
  void inferOutputTypes();
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_FULLY_CONNECTED_OP_H_
