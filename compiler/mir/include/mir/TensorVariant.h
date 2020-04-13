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

#ifndef _MIR_TENSOR_VARIANT_H_
#define _MIR_TENSOR_VARIANT_H_

#include "mir/Common.h"
#include "mir/Index.h"
#include "mir/TensorType.h"

#include <adtidas/SmallVector.h>

#include <cassert>
#include <memory>

namespace mir
{

class TensorVariant
{
public:
  explicit TensorVariant(const TensorType &type);

  TensorVariant(const TensorType &type, const void *data);

  // TODO Remove as deprecated.
  TensorVariant(DataType element_type, const Shape &shape);

  // TODO Remove as deprecated.
  TensorVariant(DataType element_type, const Shape &shape, const void *data);

  TensorVariant(const TensorVariant &t_old, const Shape &shape);

  virtual ~TensorVariant() = default;

  char *at(const Index &idx) const { return _data.get() + getOffset(idx) * _element_size; }

  char *atOffset(int32_t offset) const
  {
    assert(offset >= 0 && offset < getShape().numElements());
    return _data.get() + offset * _element_size;
  }

  size_t getOffset(const Index &idx) const
  {
    assert(idx.rank() == getShape().rank());
    std::size_t offset = 0;
    for (int i = 0; i < getShape().rank(); ++i)
      offset += idx.at(i) * _strides[i];
    return offset;
  }

  const TensorType &getType() const { return _type; }

  DataType getElementType() const { return _type.getElementType(); }
  const Shape &getShape() const { return _type.getShape(); }

  // TODO Replace uses with `getElementType` and remove.
  DataType getDataType() const { return _type.getElementType(); }
  // FIXME This should not be a member of this class.
  size_t getElementSize() const { return _element_size; }

private:
  TensorType _type;
  std::shared_ptr<char> _data;
  adt::small_vector<int_fast32_t, MAX_DIMENSION_COUNT> _strides;

  size_t _element_size;
};

} // namespace mir

#endif //_MIR_TENSOR_VARIANT_H_
