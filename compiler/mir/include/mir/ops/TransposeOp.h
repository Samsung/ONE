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

#ifndef _MIR_OPS_TRANSPOSE_OP_H_
#define _MIR_OPS_TRANSPOSE_OP_H_

#include "mir/Operation.h"
#include <vector>

namespace mir
{
namespace ops
{

/**
 * @brief Tensor transpose operation.
 *
 * Rearranges axes of input tensor.
 */
class TransposeOp : public Operation
{
public:
  TransposeOp(Output *arg, const std::vector<std::size_t> &axis_order);

  const std::vector<std::size_t> &getAxisOrder() const { return _axis_order; }

  Operation *copyWithInputs(const std::vector<Output *> &arg) override
  {
    return new TransposeOp(arg[0], _axis_order);
  }

private:
  void inferOutputTypes();

  std::vector<std::size_t> _axis_order;
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_TRANSPOSE_OP_H_
