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

#include "ConcatSpec.h"

#include <cassert>

using namespace nncc::core::ADT::tensor;

nncc::core::ADT::tensor::Shape ConcatSpec::forward(const ShapeList &inputs) const
{
  assert(inputs.size() > 0);

  Shape output_shape = inputs.at(0);

  for (uint32_t n = 1; n < inputs.size(); ++n)
  {
    // The current implementation assumes that "inputs" is well-formed
    // TODO Verify whether "inputs" is really well-formed
    const auto &input_shape = inputs.at(n);
    output_shape.dim(_axis) += input_shape.dim(_axis);
  }

  return output_shape;
}

ConcatSpec concat_spec(uint32_t axis) { return ConcatSpec{axis}; }
