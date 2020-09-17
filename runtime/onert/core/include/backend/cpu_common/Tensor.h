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

class DynamicMemoryManager;

class Tensor : public IPortableTensor
{
public:
  Tensor() = delete;
  virtual ~Tensor();

public:
  Tensor(const ir::OperandInfo &info, const ir::Layout layout,
         DynamicMemoryManager *dynamic_mem_mgr)
      : _info(info), _layout(layout), _buffer(nullptr), _num_references(0),
        _dynamic_mem_mgr(dynamic_mem_mgr), _allocator(nullptr)
  {
    // DO NOTHING
  }

public:
  // Only one of two method 'setBuffer' must be called once

  /**
   * @brief Set the Buffer object. This method is called for static and non-const tensor
   */
  void setBuffer(uint8_t *buffer)
  {
    assert(_buffer == nullptr);
    _buffer = buffer;
  }

  /**
   * @brief Set the Buffer object. This method is called for dynamic or const tensor
   */
  void setBuffer(const std::shared_ptr<Allocator> &alloc)
  {
    assert(_buffer == nullptr);
    _allocator = alloc;
    _buffer = alloc->base();
  }

  // This works just as setBuffer but it simply overwrite existing Allocator without nullptr check
  void overwriteBuffer(const std::shared_ptr<Allocator> &alloc)
  {
    _allocator = alloc;
    _buffer = alloc->base();
  }

  /**
   * @brief Mark this tensor does not have memory.
   *        Real memory deallocation should be done by caller.
   */
  void resetBuffer()
  {
    _allocator.reset();
    _buffer = nullptr;
  }

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
  size_t dimension(size_t index) const override { return _info.shape().dim(index); }
  size_t num_dimensions() const override { return _info.shape().rank(); }
  size_t total_size() const override { return _info.total_size(); }
  size_t calcOffset(const ir::Coordinates &coords) const override;
  ir::Layout layout() const override { return _layout; }
  ir::DataType data_type() const override { return _info.typeInfo().type(); }
  float data_scale() const override { return _info.typeInfo().scale(); }
  int32_t data_offset() const override { return _info.typeInfo().offset(); }
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

  void setShape(const ir::Shape &new_shape) override;

protected:
  ir::OperandInfo _info;
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

} // namespace cpu_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_COMMON_TENSOR_H__
