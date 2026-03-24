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

#include "mir/ops/ConcatOp.h"

namespace mir
{
namespace ops
{

void ConcatOp::inferOutputTypes()
{
  Shape output_shape(getInputShape(0));
  output_shape.dim(_axis) = 0;
  auto element_type = getInput(0)->getElementType();

  for (std::size_t i = 0; i < getNumInputs(); ++i)
  {
    output_shape.dim(_axis) += getInputShape(i).dim(_axis);
    assert(getInput(i)->getElementType() == element_type);
  }

  setOutputType(0, {element_type, output_shape});
}

} // namespace ops
} // namespace mir
