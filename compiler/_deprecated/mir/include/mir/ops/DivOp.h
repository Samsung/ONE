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

#ifndef _MIR_OPS_DIV_OP_H_
#define _MIR_OPS_DIV_OP_H_

#include "mir/ops/BinaryElementwiseOp.h"

namespace mir
{
namespace ops
{

class DivOp : public BinaryElementwiseOp
{
public:
  DivOp(Output *arg1, Output *arg2) : BinaryElementwiseOp(Type::div, arg1, arg2) {}

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    return new DivOp(inputs[0], inputs[1]);
  }
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_DIV_OP_H_
