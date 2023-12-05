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

#ifndef __ONERT_BACKEND_BASIC_TENSOR_H__
#define __ONERT_BACKEND_BASIC_TENSOR_H__

#include "Allocator.h"

#include <backend/IPortableTensor.h>
#include <ir/OperandInfo.h>
#include <ir/Data.h>

namespace onert
{
namespace backend
{
namespace basic
{

class DynamicMemoryManager;

class Tensor : public IPortableTensor
{
public:
  Tensor() = delete;
  virtual ~Tensor();

public:
  Tensor(const ir::OperandInfo &info, const ir::Layout layout,
         DynamicMemoryManager *dynamic_mem_mgr)
    : IPortableTensor(info), _layout(layout), _buffer(nullptr), _num_references(0),
      _dynamic_mem_mgr(dynamic_mem_mgr), _allocator(nullptr)
  {
    // DO NOTHING
  }

public:
  // Only one of two method 'setBuffer' must be called once

  /**
   * @brief Set the Buffer object. This method is called for static and non-const tensor
   */
  void setBuffer(uint8_t *buffer) { _buffer = buffer; }

  /**
   * @brief Set the Buffer object. This method is called for dynamic or const tensor
   */
  void setBuffer(const std::shared_ptr<Allocator> &alloc)
  {
    _allocator = alloc;
    _buffer = alloc->base();
  }

  /**
   * @brief Reset the buffer and deallocate the allocation if it is managed by itself
   */
  void deallocBuffer() override;

public:
  uint8_t *buffer() const override { return _buffer; }
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
  size_t total_size() const override { return _info.total_size(); }
  size_t calcOffset(const ir::Coordinates &coords) const override;
  ir::Layout layout() const override { return _layout; }
  ir::DataType data_type() const override { return _info.typeInfo().type(); }
  bool is_constant() const override { return _info.isConstant(); }
  bool is_dynamic() const override { return _info.isDynamic(); }
  void set_dynamic() override { _info.setDynamic(); }
  bool applyShape(const ir::Shape &new_shape) override;
  const ir::Sparsity *sparsity() const override { return _info.typeInfo().sparsity(); }

  virtual void increase_ref()
  {
    assert(is_dynamic() ||
           // when not dynamic
           (_buffer != nullptr));

    ++_num_references;
  }

  virtual void decrease_ref()
  {
    assert(_buffer != nullptr || _allocator != nullptr);
    assert(_num_references > 0);
    --_num_references;
    // constant tensor and dynamic tensor has _allocator
    if (_num_references == 0)
    {
      if (_buffer != nullptr)
        _buffer = nullptr;
      if (_allocator != nullptr)
      {
        _allocator->release();
        _allocator = nullptr;
      }
    }
  }

  /**
   * @brief Reset reference count to zero and release data
   */
  virtual void reset_ref()
  {
    assert(_buffer != nullptr || _allocator != nullptr);
    assert(_num_references > 0);
    _num_references = 0;

    // Only constant tensor has allocator pointer
    if (_buffer != nullptr)
      _buffer = nullptr;
    else
    {
      _allocator->release();
      _allocator = nullptr;
    }
  }

  virtual int32_t num_references() { return _num_references; }

  void setShape(const ir::Shape &new_shape) override;
  ir::Shape getShape() const override;

protected:
  ir::Layout _layout;
  uint8_t *_buffer;
  int32_t _num_references;
  DynamicMemoryManager *_dynamic_mem_mgr;

private:
  /**
   * @brief Memory allocator for dynamic tensor and const tensor
   *        Since maintaing _allocator and also _buffer makes confusion,
   *        we will mainly use _buffer (not _allocator.base()) for memory pointer in this code.
   *        _allocator(shared_ptr) is used to guarantee that we have valid _buffer.
   */
  std::shared_ptr<Allocator> _allocator;
};

/**
 * @brief Class that uses data from external memory that is not managed by a backend
 *        instead of allocating and copying the data. ExternalTensor's data pointer points to
 *        an address of memory such as where memory is already allocated, or mmapped area.
 *        This is meaning that ExternalTensor can take all of types' ir::Data.
 *        To support this, assume below things no padding, always NHWC layout,
 *        constant tensor and not dynamic.
 */
class ExternalTensor : public Tensor
{
public:
  ExternalTensor() = delete;
  virtual ~ExternalTensor();

public:
  ExternalTensor(const ir::OperandInfo &info, const ir::Layout layout)
    : Tensor(info, layout, nullptr)
  {
    assert(_layout == ir::Layout::NHWC || _layout == ir::Layout::UNKNOWN);
    assert(_info.isConstant());
    assert(_info.isDynamic() == false);
  }

public:
  /**
   * @brief     set Data to be shared from external so that this ExternalTensor will not be
   *            allocated on CPU backend
   * @param[in] data    data of Operand to be set
   */
  void setData(const std::shared_ptr<ir::Data> data)
  {
    assert(data != nullptr);
    _data = data;
    // Note. Some op such as cker::Conv could take buffer as nullptr.
    // That's why _buffer also would be used
    _buffer = const_cast<uint8_t *>(_data->base());
  }

public:
  uint8_t *buffer() const override { return _buffer; }

  bool is_constant() const override { return true; }
  bool is_dynamic() const override { return false; }
  void set_dynamic() override
  {
    throw std::runtime_error("This tensor does not support changing dynamic");
  }

  void setShape(const ir::Shape &) override
  {
    throw std::runtime_error("This tensor does not support changing shape");
  }

  void increase_ref() override { ++_num_references; }

  void decrease_ref() override
  {
    assert(_data != nullptr);
    assert(_num_references > 0);
    --_num_references;
    if (_num_references == 0)
    {
      _data.reset();
      _buffer = nullptr;
    }
  }

  /**
   * @brief Reset reference count to zero and release data
   */
  void reset_ref() override
  {
    assert(_data != nullptr);
    assert(_num_references > 0);
    _num_references = 0;

    _data.reset();
    _buffer = nullptr;
  }

  int32_t num_references() override { return _num_references; }

private:
  std::shared_ptr<const ir::Data> _data;
};
} // namespace basic
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BASIC_TENSOR_H__
