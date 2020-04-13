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

#ifndef _MIR_OPS_RESHAPE_OP_H_
#define _MIR_OPS_RESHAPE_OP_H_

#include "mir/Operation.h"

namespace mir
{
namespace ops
{

class ReshapeOp : public Operation
{
public:
  ReshapeOp(Output *arg, const Shape &shape) : Operation(Type::reshape, {arg})
  {
    const Shape &input_shape = getInputShape(0);
    auto output_shape = shape;

    auto in_elements_num = input_shape.numElements();
    int32_t out_elements_num = 1;
    // Can't use num_elements due to -1 in input shape and Shape using unsigned ints for dimensions.
    for (int32_t d = 0; d < output_shape.rank(); ++d)
    {
      auto dim = output_shape.dim(d);
      if (dim != Shape::autoDim)
        out_elements_num *= dim;
    }

    for (int32_t d = 0; d < output_shape.rank(); ++d)
    {
      auto &dim = output_shape.dim(d);
      if (dim == Shape::autoDim)
        dim = static_cast<int32_t>(in_elements_num / out_elements_num);
    }

    setOutputType(0, {arg->getElementType(), output_shape});
  }

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    return new ReshapeOp(inputs[0], getOutputShape(0));
  }
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_RESHAPE_OP_H_
