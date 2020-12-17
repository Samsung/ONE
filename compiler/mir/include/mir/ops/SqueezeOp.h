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

#ifndef _MIR_OPS_SQUEEZE_OP_H_
#define _MIR_OPS_SQUEEZE_OP_H_

#include "mir/Operation.h"
#include <algorithm>

namespace mir
{
namespace ops
{

class SqueezeOp : public Operation
{
public:
  SqueezeOp(Output *arg, const std::vector<std::int32_t> &dims_to_squeeze)
    : Operation(Type::squeeze, {arg}), _dims_to_squeeze(dims_to_squeeze)
  {
    // Infer output shape.
    inferOutputTypes();
  }

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    return new SqueezeOp(inputs[0], _dims_to_squeeze);
  }

  void inferOutputTypes();

  int32_t getNumSqueezeDims() const { return static_cast<int32_t>(_dims_to_squeeze.size()); }

  const std::vector<int32_t> &getDimsToSqueeze() const { return _dims_to_squeeze; }

private:
  std::vector<int32_t> _dims_to_squeeze;
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_SQUEEZE_OP_H_
