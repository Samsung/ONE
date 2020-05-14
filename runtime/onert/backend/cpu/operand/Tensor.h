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

#ifndef __ONERT_BACKEND_CPU_OPERAND_TENSOR_H__
#define __ONERT_BACKEND_CPU_OPERAND_TENSOR_H__

#include "Allocator.h"

#include <backend/ITensor.h>
#include <ir/OperandInfo.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace operand
{

class Tensor : public ITensor
{
public:
  Tensor() = delete;

public:
  Tensor(const ir::OperandInfo &info)
      : _info(info), _buffer(nullptr), _num_references(0), _allocator(nullptr)
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
  void setBuffer(const std::shared_ptr<cpu_common::Allocator> &alloc)
  {
    assert(_buffer == nullptr && _allocator == nullptr);
    _allocator = alloc;
  }
  float scale() const { return _info.typeInfo().scale(); }
  int32_t offset() const { return _info.typeInfo().offset(); }

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
  ir::Layout layout() const override { return ir::Layout::NHWC; }
  ir::DataType data_type() const override { return _info.typeInfo().type(); }
  float data_scale() const { return _info.typeInfo().scale(); }
  int32_t data_offset() const { return _info.typeInfo().offset(); }
  bool has_padding() const override { return false; }
  void access(const std::function<void(ITensor &tensor)> &fn) final;
  bool is_dynamic() const override { return _info.memAllocType() == ir::MemAllocType::DYNAMIC; }

  void increase_ref()
  {
    assert(_buffer != nullptr || _allocator != nullptr);
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

  void dimension(size_t index, size_t dim) override
  {
    if (!(index < static_cast<size_t>(_info.shape().rank())))
    {
      throw std::runtime_error("index should be less than rank");
    }

    _info.shape().dim(index) = dim;
  }

  void num_dimensions(size_t rank) override
  {
    ir::Shape new_shape(rank); // all dims are initialized to 0 (invalid dim)
    _info.shape(new_shape);
  };

private:
  ir::OperandInfo _info;
  uint8_t *_buffer;
  int32_t _num_references;
  std::shared_ptr<cpu_common::Allocator> _allocator;
};

} // namespace operand
} // namespace cpu
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_OPERAND_TENSOR_H__
