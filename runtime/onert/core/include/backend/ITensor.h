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

#ifndef __ONERT_BACKEND_OPERAND_I_TENSOR_H__
#define __ONERT_BACKEND_OPERAND_I_TENSOR_H__

#include <cstring>
#include <cstdint>
#include <functional>

#include "ir/DataType.h"
#include "ir/Layout.h"
#include "ir/Shape.h"
#include "ir/Coordinates.h"
#include "ir/OperandInfo.h"

namespace onert
{
namespace backend
{

class ITensor
{
public:
  virtual ~ITensor() = default;

public:
  virtual uint8_t *buffer() const = 0;
  virtual size_t total_size() const = 0;
  virtual size_t dimension(size_t index) const = 0;
  virtual size_t num_dimensions() const = 0;
  virtual size_t calcOffset(const ir::Coordinates &coords) const = 0;
  virtual ir::Layout layout() const = 0;
  virtual ir::DataType data_type() const = 0;
  virtual float data_scale() const { return 0; }    // XXX Remove default impl
  virtual int32_t data_offset() const { return 0; } // XXX Remove default impl
  virtual bool has_padding() const = 0;
  virtual void access(const std::function<void(ITensor &tensor)> &fn) = 0;

  /**
   * @brief Return true if the tensor needs dynamic allocation, meaning that during compile-time
   *        the outpus shape cannot be known and the output shape is calculated during
   *        kernel execution-time.
   */
  virtual bool is_dynamic() const
  {
    throw std::runtime_error("This backend does not support dynamic tensor");
  }

  /// @brief set this tensor dynamic
  virtual void set_dynamic()
  {
    throw std::runtime_error("This backend does not support dynamic tensor");
  }

  // set dim when this tensor is dynamic
  virtual void dimension(size_t /* index */, size_t /* dim */)
  {
    throw std::runtime_error("This backend does not support dynamic tensor");
  }

  // set the rank when this tensor is dynamic.
  virtual void num_dimensions(size_t /*rank*/)
  {
    throw std::runtime_error("This backend does not support dynamic tensor");
  }
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

/**
 * @brief Set the shape of tenser to new_shape
 */
void setShape(ITensor *tensor, const ir::Shape &new_shape);

/**
 * @brief Get ir::Shape of tensor
 * @note  Higer dimension will be placed on front.
 */
ir::Shape getShape(ITensor *tensor);

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_OPERAND_I_TENSOR_H__
