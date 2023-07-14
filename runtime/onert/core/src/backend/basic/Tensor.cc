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

#include "backend/basic/Tensor.h"

#include "ir/DataType.h"
#include "backend/basic/MemoryManager.h"

namespace onert
{
namespace backend
{
namespace basic
{

Tensor::~Tensor() {}

size_t Tensor::calcOffset(const ir::Coordinates &coords) const
{
  auto shape = getShape();
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

void Tensor::setShape(const ir::Shape &new_shape) { _info.shape(new_shape); }

bool Tensor::applyShape(const ir::Shape &new_shape)
{
  bool previously_dynamic = is_dynamic();

  auto allocTensorMem = [&]() {
    auto capacity = total_size();
    assert(_dynamic_mem_mgr);
    auto alloc = _dynamic_mem_mgr->allocate(this, capacity);
    setBuffer(alloc);
  };

  if (!previously_dynamic || buffer() == nullptr)
  {
    // Always set shape - when buffer with same size was already allocated, shape could differ
    setShape(new_shape);
    set_dynamic();
    allocTensorMem();
  }
  else
  {
    auto previous_size = total_size();
    auto new_size = new_shape.num_elements() * ir::sizeOfDataType(data_type());
    if (previous_size != new_size)
    {
      assert(_dynamic_mem_mgr);
      _dynamic_mem_mgr->deallocate(this);

      setShape(new_shape);
      set_dynamic();
      allocTensorMem();
    }
    else
    { // when buffer with same size was already allocated, shape could differ
      setShape(new_shape);
    }
  }
  return true;
}

ir::Shape Tensor::getShape() const { return _info.shape(); }

void Tensor::deallocBuffer()
{
  if (_allocator)
  {
    _buffer = nullptr;
    _allocator.reset();
    if (_dynamic_mem_mgr)
    {
      _dynamic_mem_mgr->deallocate(this);
    }
  }
}

} // namespace basic
} // namespace backend
} // namespace onert

// ExternalTensor

namespace onert
{
namespace backend
{
namespace basic
{

// `dynamic_cast` not working across library boundaries on NDK
// With this as a key function, `dynamic_cast` works across dl
ExternalTensor::~ExternalTensor() {}

} // namespace basic
} // namespace backend
} // namespace onert
