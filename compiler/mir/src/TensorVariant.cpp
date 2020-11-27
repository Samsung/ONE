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

#include "mir/TensorVariant.h"
#include <cstring>

namespace mir
{

TensorVariant::TensorVariant(const TensorType &type) : _type(type), _strides(type.getShape().rank())
{
  _element_size = getDataTypeSize(getElementType());
  std::size_t data_size = getShape().numElements() * _element_size;
  _data.reset(new char[data_size], std::default_delete<char[]>());

  int stride = 1;
  for (int d = getShape().rank() - 1; d >= 0; --d)
  {
    _strides[d] = stride;
    stride *= getShape().dim(d);
  }
}

TensorVariant::TensorVariant(DataType element_type, const Shape &shape)
  : TensorVariant(TensorType(element_type, shape))
{
}

TensorVariant::TensorVariant(const TensorType &type, const void *data) : TensorVariant(type)
{
  std::size_t data_size = getShape().numElements() * _element_size;
  std::memcpy(_data.get(), data, data_size);
}

TensorVariant::TensorVariant(DataType element_type, const Shape &shape, const void *data)
  : TensorVariant(TensorType(element_type, shape), data)
{
}

/**
 * @brief Construct a TensorVariant from t_old that has strides with 0 where dim = 1
 * Used for broadcasting
 * @param t_old TensorVariant to use as base
 * @param shape shape to broadcast to
 */
TensorVariant::TensorVariant(const TensorVariant &t_old, const Shape &shape)
  : _type(t_old.getType().getElementType(), shape), _data(t_old._data),
    _strides(static_cast<size_t>(shape.rank())), _element_size(t_old._element_size)
{
  int axis_old = t_old.getShape().rank() - 1;
  for (int d = shape.rank() - 1; d >= 0; d--)
  {
    if (axis_old == -1)
      break;
    if (t_old.getShape().dim(axis_old) != 1)
      _strides[d] = t_old._strides[axis_old];
    axis_old--;
  }
}

} // namespace mir
