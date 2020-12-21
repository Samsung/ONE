/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "UserTensor.h"

#include "util/Exceptions.h"
#include "ir/DataType.h"

namespace onert
{
namespace backend
{
namespace builtin
{

size_t UserTensor::calcOffset(const ir::Coordinates &coords) const
{
  size_t rank = getShape().rank();
  size_t offset = 0;
  for (size_t i = 0; i < rank; ++i)
  {
    offset = offset * getShape().dim(i) + coords[i];
  }
  offset *= sizeOfDataType(data_type());
  return offset;
}

bool UserTensor::applyShape(const ir::Shape &new_shape)
{
  // User tensors cannot be reallocated.
  auto new_size = new_shape.num_elements() * ir::sizeOfDataType(data_type());
  if (total_size() < new_size)
    throw InsufficientBufferSizeException{"User given buffer size is too small."};
  setShape(new_shape);
  return true;
}

} // namespace builtin
} // namespace backend
} // namespace onert
