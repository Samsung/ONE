/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "backend/cpu_common/Tensor.h"

namespace onert
{
namespace backend
{
namespace cpu_common
{

size_t Tensor::calcOffset(const ir::Coordinates &coords) const
{
  size_t rank = num_dimensions();
  rank = rank == 0 ? 1 : rank;
  size_t offset = 0;
  for (size_t i = 0; i < rank; ++i)
  {
    offset = offset * dimension(i) + coords[i];
  }
  offset *= sizeOfDataType(data_type());
  return offset;
}

void Tensor::setShape(const ir::Shape &new_shape) { _info.shape(new_shape); }

} // namespace cpu_common
} // namespace backend
} // namespace onert
