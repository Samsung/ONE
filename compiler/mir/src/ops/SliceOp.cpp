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

#include "mir/ops/SliceOp.h"

namespace mir
{
namespace ops
{

// Only supports 4d inputs
void SliceOp::inferOutputTypes()
{
  const Shape &input_shape = getInputShape(0);
  assert(input_shape.rank() <= 4 && "Support only 4D tensors or smaller");
  Shape output_shape(input_shape.rank());
  for (int i = 0; i < input_shape.rank(); i++)
  {
    if (_sizes.dim(i) == -1)
    {
      output_shape.dim(i) = input_shape.dim(i) - _starts.dim(i);
    }
    else
    {
      output_shape.dim(i) = _sizes.dim(i);
    }
  }
  setOutputType(0, {getInput(0)->getElementType(), output_shape});
}

} // namespace ops
} // namespace mir
