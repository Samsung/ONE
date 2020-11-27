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

#ifndef _MIR_OPS_RESIZE_OP_H_
#define _MIR_OPS_RESIZE_OP_H_

#include "mir/Operation.h"
#include "mir/Shape.h"
#include <vector>
#include <cmath>

namespace mir
{
namespace ops
{

/**@brief Resize operation
 * scales are such that output = input * scale for each dimension
 * and the number of dimensions matches
 */
class ResizeOp : public Operation
{
public:
  enum class ResizeMethod
  {
    nearestNeighbor, // TODO: BICUBIC and BILINEAR
  };

  ResizeOp(Output *arg, ResizeMethod mode, const std::vector<float> &scales)
    : Operation(Type::resizeIm, {arg}), _mode(mode), _scales(scales)
  {
    // Infer output shape based on given scales.
    auto &input_shape = getInputShape(0);
    assert(input_shape.rank() == 4 && _scales.size() == 4);
    Shape output_shape(input_shape.rank());

    for (int32_t i = 0; i < input_shape.rank(); ++i)
    {
      output_shape.dim(i) = static_cast<int32_t>(lroundf(_scales.at(i) * input_shape.dim(i)));
    }

    setOutputType(0, {getInput(0)->getElementType(), output_shape});
  }

  Operation *copyWithInputs(const std::vector<Output *> &inputs) override
  {
    return new ResizeOp(inputs[0], _mode, getOutputShape(0));
  }

  ResizeOp(Output *arg, ResizeMethod mode, const Shape &output_shape)
    : Operation(Type::resizeIm, {arg}), _mode(mode)
  {
    // Calculate scales based on given shape.
    auto &input_shape = getInputShape(0);
    assert(input_shape.rank() == 4 && output_shape.rank() == 4);
    setOutputType(0, {getInput(0)->getElementType(), output_shape});
    _scales = {1.0f, static_cast<float>(output_shape.dim(1)) / input_shape.dim(1),
               static_cast<float>(output_shape.dim(2)) / input_shape.dim(2), 1.0f};
  }

  /** @return The resize mode */
  ResizeMethod getMode() const { return _mode; }

  const std::vector<float> &getScales() const { return _scales; }

private:
  std::vector<float> _scales;
  ResizeMethod _mode;
};

} // namespace ops
} // namespace mir

#endif //_MIR_OPS_RESIZE_OP_H_
