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

#ifndef _MIR_OPS_LEAKY_RELU_OP_H_
#define _MIR_OPS_LEAKY_RELU_OP_H_

#include "mir/Operation.h"

namespace mir
{
namespace ops
{

class LeakyReluOp : public Operation
{
public:
  explicit LeakyReluOp(Output *arg, float alpha) : Operation(Type::leakyReLU, {arg}), _alpha(alpha)
  {
    // Infer output shape.
    setOutputType(0, {arg->getElementType(), arg->getShape()});
  }

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    return new LeakyReluOp(inputs[0], _alpha);
  }

  float getAlpha() const { return _alpha; }

private:
  float _alpha;
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_LEAKY_RELU_OP_H_
