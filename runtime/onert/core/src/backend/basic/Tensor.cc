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

#include "backend/basic/MemoryManager.h"
#include "ir/DataType.h"

namespace onert::backend::basic
{

Tensor::~Tensor() {}

// Initialize _size to 0 because actual buffer size may be unknown until inference for dynamic
// shapes including unknown dimensions
// Otherwise, it should be initialized to total size of operand info
Tensor::Tensor(const ir::OperandInfo &info, DynamicMemoryManager *dynamic_mem_mgr)
  : IPortableTensor(info), _buffer(nullptr),
    _size(info.shape().hasUnspecifiedDims() ? 0 : info.total_size()), _num_references(0),
    _dynamic_mem_mgr(dynamic_mem_mgr), _allocator(nullptr)
{
  // DO NOTHING
}

void Tensor::setShape(const ir::Shape &new_shape) { _info.shape(new_shape); }

bool Tensor::applyShape(const ir::Shape &new_shape)
{
  if (_buffer != nullptr && new_shape == _info.shape())
    return true;

  // Always set shape - when buffer with same or larger size was already allocated, shape could
  // differ
  _info.shape(new_shape);
  set_dynamic();
  if (_buffer == nullptr || _size < _info.total_size())
  {
    assert(_dynamic_mem_mgr);
    if (_allocator)
      _dynamic_mem_mgr->deallocate(this);

    _size = _info.total_size();
    setBuffer(_dynamic_mem_mgr->allocate(this, _size));
  }

  return true;
}

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

} // namespace onert::backend::basic

// ExternalTensor

namespace onert::backend::basic
{

// `dynamic_cast` not working across library boundaries on NDK
// With this as a key function, `dynamic_cast` works across dl
ExternalTensor::~ExternalTensor() {}

} // namespace onert::backend::basic
