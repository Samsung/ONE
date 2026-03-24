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

#ifndef _MIR_OPS_SLICE_OP_H_
#define _MIR_OPS_SLICE_OP_H_

#include "mir/Operation.h"

namespace mir
{
namespace ops
{

class SliceOp : public Operation
{
public:
  SliceOp(Output *arg, const Shape &starts, const Shape &sizes)
    : Operation(Type::slice, {arg}), _starts(starts), _sizes(sizes)
  {
    inferOutputTypes();
  }

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    return new SliceOp(inputs[0], _starts, _sizes);
  }

  const Shape &getStarts() { return _starts; }

  const Shape &getSizes() { return _sizes; }

private:
  void inferOutputTypes();

  Shape _starts;
  Shape _sizes;
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_SLICE_OP_H_
