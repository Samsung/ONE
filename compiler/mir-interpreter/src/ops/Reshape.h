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

#ifndef _NNC_CORE_BACKEND_INTERPRETER_RESHAPE_IMPL_
#define _NNC_CORE_BACKEND_INTERPRETER_RESHAPE_IMPL_

#include "mir/ShapeRange.h"
#include "mir/TensorVariant.h"

#include <cstring>

namespace mir_interpreter
{

std::vector<mir::TensorVariant> Reshape(const mir::TensorVariant &input,
                                        const mir::Shape &output_shape)
{
  assert(input.getShape().numElements() == output_shape.numElements());
  mir::TensorType type(input.getElementType(), output_shape);
  if (input.getType().isQuantized())
    type.setQuantization(input.getType().getQuantization());

  mir::TensorVariant result(type);
  mir::ShapeRange input_range(input.getShape());
  auto in_iter = input_range.begin();
  const size_t elem_size = input.getElementSize();

  for (const auto &out_index : mir::ShapeRange(output_shape))
    std::memcpy(result.at(out_index), input.at(*in_iter++), elem_size);

  return {result};
}

} // namespace mir_interpreter

#endif //_NNC_CORE_BACKEND_INTERPRETER_RESHAPE_IMPL_
