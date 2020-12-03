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

#include "ir/DataType.h"
#include "backend/cpu_common/MemoryManager.h"

namespace onert
{
namespace backend
{
namespace cpu_common
{

Tensor::~Tensor() {}

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

bool Tensor::applyShape(const ir::Shape &new_shape)
{
  bool previously_dynamic = is_dynamic();

  auto allocTensorMem = [&](bool overwrite = false) {
    auto capacity = total_size();
    auto alloc = _dynamic_mem_mgr->allocate(this, capacity);

    if (overwrite)
      overwriteBuffer(alloc);
    else
      setBuffer(alloc);
  };

  if (!previously_dynamic)
  {
    // TODO deallocate tensor->buffer()
    // issue is that staticTensorManager might have allocate this memory
    setShape(new_shape);
    set_dynamic();
    allocTensorMem(true);
  }
  else if (buffer() == nullptr)
  {
    setShape(new_shape);
    set_dynamic();
    allocTensorMem();
  }
  // when buffer was already allocated and new_shape requires different size
  else
  {
    auto previous_size = total_size();
    auto new_size = new_shape.num_elements() * ir::sizeOfDataType(data_type());
    if (previous_size != new_size)
    {
      _dynamic_mem_mgr->deallocate(this);

      setShape(new_shape);
      set_dynamic();
      allocTensorMem(true);
    }
    else
    { // when buffer with same size was already allocated, shape could differ
      setShape(new_shape);
    }
  }
  return true;
}

} // namespace cpu_common
} // namespace backend
} // namespace onert

// ExternalTensor

namespace onert
{
namespace backend
{
namespace cpu_common
{

// `dynamic_cast` not working across library boundaries on NDK
// With this as a key function, `dynamic_cast` works across dl
ExternalTensor::~ExternalTensor() {}

} // namespace cpu
} // namespace backend
} // namespace onert
