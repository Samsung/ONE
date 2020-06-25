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

#ifndef __ONERT_BACKEND_CPU_COMMON_TENSOR_H__
#define __ONERT_BACKEND_CPU_COMMON_TENSOR_H__

#include "Allocator.h"

#include <backend/IPortableTensor.h>
#include <ir/OperandInfo.h>

namespace onert
{
namespace backend
{
namespace cpu_common
{

class Tensor : public IPortableTensor
{
public:
  Tensor() = delete;

public:
  Tensor(const ir::OperandInfo &info, const ir::Layout layout)
      : _info(info), _layout(layout), _buffer(nullptr), _num_references(0), _allocator(nullptr)
  {
    // DO NOTHING
  }

public:
  // Only one of two method 'setBuffer' must be called once
  void setBuffer(uint8_t *buffer)
  {
    assert(_buffer == nullptr && _allocator == nullptr);
    _buffer = buffer;
  }
  void setBuffer(const std::shared_ptr<Allocator> &alloc)
  {
    assert(_buffer == nullptr && _allocator == nullptr);
    _allocator = alloc;
  }

  // This works just as setBuffer but it simply overwrite existing Allocator without nullptr check
  void overwriteBuffer(const std::shared_ptr<Allocator> &alloc) { _allocator = alloc; }

public:
  uint8_t *buffer() const override
  {
    if (_allocator != nullptr)
      return _allocator->base();
    else
      return _buffer;
  }
  /**
   * @brief Get dimension by index
   *
   * @param index Index to get diemension
   * @return size_t Dimension at index
   * @note N : dimension(0)
   *       H : dimension(1)
   *       W : dimension(2)
   *       C : dimension(3)
   */
  size_t dimension(size_t index) const override { return _info.shape().dim(index); }
  size_t num_dimensions() const override { return _info.shape().rank(); }
  size_t total_size() const override { return _info.total_size(); }
  size_t calcOffset(const ir::Coordinates &coords) const override;
  ir::Layout layout() const override { return _layout; }
  ir::DataType data_type() const override { return _info.typeInfo().type(); }
  float data_scale() const override { return _info.typeInfo().scale(); }
  int32_t data_offset() const override { return _info.typeInfo().offset(); }
  bool is_dynamic() const override { return _info.isDynamic(); }
  void set_dynamic() override { _info.setDynamic(); }

  void increase_ref()
  {
    assert(is_dynamic() ||
           // when not dynamic
           (_buffer != nullptr || _allocator != nullptr));

    ++_num_references;
  }
  void decrease_ref()
  {
    assert(_buffer != nullptr || _allocator != nullptr);
    assert(_num_references > 0);
    --_num_references;
    // Only constant tensor has allocator pointer
    if (_num_references == 0)
    {
      if (_buffer != nullptr)
        _buffer = nullptr;
      else
      {
        _allocator->release();
        _allocator = nullptr;
      }
    }
  }

  void setShape(const ir::Shape &new_shape) override;

private:
  ir::OperandInfo _info;
  ir::Layout _layout;
  uint8_t *_buffer;
  int32_t _num_references;
  std::shared_ptr<Allocator> _allocator;
};

} // namespace cpu_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_COMMON_TENSOR_H__
