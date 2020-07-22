/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CONTROLFLOW_USER_TENSOR_H__
#define __ONERT_BACKEND_CONTROLFLOW_USER_TENSOR_H__

#include "ir/OperandInfo.h"
#include "backend/IPortableTensor.h"

namespace onert
{
namespace backend
{
namespace controlflow
{

/**
 * @brief Tensor object that is for Input and Output tensors from the user.
 *
 * This class is a wrapped buffer that is allocated by the user. So it does not have resposibility
 * on allocation nor deallocation. All the model input/output tensors are wrapped with this class
 * for execution.
 *
 */
class UserTensor : public IPortableTensor
{
public:
  UserTensor(const ir::OperandInfo &info, ir::Layout layout, uint8_t *buffer, size_t size)
      : _info{info}, _layout{layout}, _buffer{buffer}, _size{size}, _dynamic{false}
  {
  }

  UserTensor(const ir::OperandInfo &info, ir::Layout layout) : UserTensor{info, layout, nullptr, 0}
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
  size_t calcOffset(const ir::Coordinates &coords) const override;
  ir::Layout layout() const override { return _layout; }
  ir::DataType data_type() const override { return _info.typeInfo().type(); }
  float data_scale() const override { return _info.typeInfo().scale(); }
  int32_t data_offset() const override { return _info.typeInfo().offset(); }
  bool is_dynamic() const override { return _dynamic; }
  void set_dynamic() override { _dynamic = true; }
  ir::Shape getShape() const override { return _info.shape(); }
  void setShape(const ir::Shape &new_shape) override { _info.shape(new_shape); }
  bool is_constant() const override { return false; }

private:
  ir::OperandInfo _info;
  ir::Layout _layout;
  uint8_t *_buffer;
  size_t _size;
  bool _dynamic;
};

} // namespace controlflow
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CONTROLFLOW_USER_TENSOR_H__
