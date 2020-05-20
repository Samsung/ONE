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

#ifndef __ONERT_BACKEND_CONTROLFLOW_OPERAND_TENSOR_H__
#define __ONERT_BACKEND_CONTROLFLOW_OPERAND_TENSOR_H__

#include "Allocator.h"

#include <backend/ITensor.h>
#include <ir/OperandInfo.h>
#include <util/Utils.h>

namespace onert
{
namespace backend
{
namespace controlflow
{
namespace operand
{

class Tensor : public ITensor
{
public:
  Tensor() = delete;

public:
  Tensor(const ir::OperandInfo &info, const ir::Layout layout)
      : _info(info), _layout(layout), _buffer(nullptr), _allocator(nullptr)
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
    assert(_allocator == nullptr);
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
  ir::Layout layout() const override { return _layout; }
  ir::DataType data_type() const override { return _info.typeInfo().type(); }
  bool has_padding() const override { return false; }
  void access(const std::function<void(ITensor &tensor)> &fn) final;
  bool is_dynamic() const override { return false; }

private:
  ir::OperandInfo _info;
  ir::Layout _layout;
  uint8_t *_buffer;
  std::shared_ptr<Allocator> _allocator;
};

class UserTensor : public ITensor
{
public:
  UserTensor(const ir::OperandInfo &info, ir::Layout layout, uint8_t *buffer, size_t size)
      : _info{info}, _layout{layout}, _buffer{buffer}, _size{size}
  {
  }

  UserTensor(const ir::OperandInfo &info, ir::Layout layout)
      : _info{info}, _layout{layout}, _buffer{nullptr}, _size{0}
  {
  }

public:
  void setBuffer(uint8_t *buffer, size_t size)
  {
    _buffer = buffer;
    _size = size;
  }

public:
  uint8_t *buffer() const override { return _buffer; }
  size_t total_size() const override { return _size; }
  size_t dimension(size_t index) const override { return _info.shape().dim(index); }
  size_t num_dimensions() const override { return _info.shape().rank(); }
  size_t calcOffset(const ir::Coordinates &coords) const override
  {
    size_t rank = num_dimensions();
    size_t offset = 0;
    for (size_t i = 0; i < rank; ++i)
    {
      offset = offset * dimension(i) + coords[i];
    }
    offset *= sizeOfDataType(data_type());
    return offset;
  }
  ir::Layout layout() const override { return _layout; }
  ir::DataType data_type() const override { return _info.typeInfo().type(); }
  float data_scale() const override { return _info.typeInfo().scale(); }
  int32_t data_offset() const override { return _info.typeInfo().offset(); }
  bool has_padding() const override { return false; }
  bool is_dynamic() const override { return false; }
  void access(const std::function<void(backend::ITensor &tensor)> &fn) override
  {
    // This is an optional input
    if (total_size() == 0)
      return;

    fn(*this);
  }

private:
  ir::OperandInfo _info;
  ir::Layout _layout;
  uint8_t *_buffer;
  size_t _size;
};

} // namespace operand
} // namespace controlflow
} // namespace backend
} // namespace onert

#endif //  __ONERT_BACKEND_CONTROLFLOW_OPERAND_TENSOR_H__
