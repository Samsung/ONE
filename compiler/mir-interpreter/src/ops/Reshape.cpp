/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Reshape.h"

#include "mir/ShapeRange.h"

#include <cstring>

namespace mir_interpreter
{

void Reshape(const mir::TensorVariant &input, mir::TensorVariant &output)
{
  assert(input.getShape().numElements() == output.getShape().numElements());

  mir::ShapeRange input_range(input.getShape());
  auto in_iter = input_range.begin();
  const size_t elem_size = input.getElementSize();

  for (const auto &out_index : mir::ShapeRange(output.getShape()))
    std::memcpy(output.at(out_index), input.at(*in_iter++), elem_size);
}

} // namespace mir_interpreter
