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

#include "backend/IPortableTensor.h"

namespace onert::backend
{

// `dynamic_cast` not working across library boundaries on NDK
// With this as a key function, `dynamic_cast` works across dl
IPortableTensor::~IPortableTensor() {}

size_t IPortableTensor::calcOffset(const ir::Coordinates &coords) const
{
  auto shape = _info.shape();
  size_t rank = shape.rank();
  rank = rank == 0 ? 1 : rank;
  size_t offset = 0;
  for (size_t i = 0; i < rank; ++i)
  {
    auto dim = shape.rank() == 0 ? 1 : shape.dim(i);
    offset = offset * dim + coords[i];
  }
  offset *= sizeOfDataType(data_type());
  return offset;
}

} // namespace onert::backend
