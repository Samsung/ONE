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

#ifndef _MIR_TENSOR_UTIL_H_
#define _MIR_TENSOR_UTIL_H_

#include "mir/Index.h"
#include "mir/ShapeRange.h"
#include "mir/TensorVariant.h"

#include <cstring>

namespace mir
{

template <int32_t... Ints> Shape transposeShape(const Shape &shape)
{
  assert(sizeof...(Ints) == shape.rank());
  return {shape.dim(Ints)...};
}

template <unsigned int... Ints> TensorVariant transposeTensor(const TensorVariant &tensor)
{
  const auto &shape = tensor.getShape();
  Shape transposed_shape{shape.dim(Ints)...};

  auto elem_type = tensor.getElementType();
  auto elem_size = tensor.getElementSize();
  TensorType transposed_type(elem_type, transposed_shape);
  if (tensor.getType().isQuantized())
    transposed_type.setQuantization(tensor.getType().getQuantization());

  TensorVariant transposed_tensor(transposed_type);

  for (const auto &index : ShapeRange(shape))
  {
    Index transposed_index{index.at(Ints)...};
    std::memcpy(transposed_tensor.at(transposed_index), tensor.at(index), elem_size);
  }

  return transposed_tensor;
}

} // namespace mir

#endif //_MIR_TENSOR_UTIL_H_
