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

#ifndef __ONERT_BACKEND_CONTROLFLOW_INDIRECT_TENSOR_H__
#define __ONERT_BACKEND_CONTROLFLOW_INDIRECT_TENSOR_H__

#include "backend/IPortableTensor.h"
#include "UserTensor.h"

namespace onert
{
namespace backend
{
namespace controlflow
{

/**
 * @brief Tensor object that indirects to the tensor it is pointing to.
 *
 * A model I/O tensor could be two types.
 *
 * 1. @c UserTensor, if it is the primary graph
 * 2. Any other derivative of @c IPortableTensor from another backend, otherwise
 *
 * To support these, this object indirects everything to the actual tensor pointer.
 * Exceptionally if it is UserTensor, this class creates and manages it.
 */
class IOTensor : public IPortableTensor
{
public:
  IOTensor(const ir::OperandInfo &info, ir::Layout layout);
  // IOTensor(IPortableTensor *tensor);

public:
  void setTensor(IPortableTensor *tensor);
  void setUserTensor(uint8_t *buffer, size_t size);
  ir::OperandInfo orig_info() const { return _orig_info; }
  ir::Layout orig_layout() const { return _orig_layout; }

public:
  uint8_t *buffer() const override { return _tensor->buffer(); }
  size_t total_size() const override { return _tensor->total_size(); }
  size_t dimension(size_t index) const override { return _tensor->dimension(index); }
  size_t num_dimensions() const override { return _tensor->num_dimensions(); }
  size_t calcOffset(const ir::Coordinates &coords) const override
  {
    return _tensor->calcOffset(coords);
  }
  ir::Layout layout() const override { return _tensor->layout(); }
  ir::DataType data_type() const override { return _tensor->data_type(); }
  float data_scale() const override { return _tensor->data_scale(); }
  int32_t data_offset() const override { return _tensor->data_offset(); }
  bool is_dynamic() const override { return _is_dynamic; }
  void set_dynamic() override { _is_dynamic = true; }
  ir::Shape getShape() const override { return _tensor->getShape(); }
  void setShape(const ir::Shape &shape) override
  {
    // Workaround for IPortableTensor holds _info as its member
    _info.shape(shape);
    _tensor->setShape(shape);
  }
  bool is_constant() const override { return _tensor->is_constant(); }
  bool applyShape(const ir::Shape &shape) override
  {
    // Workaround for IPortableTensor holds _info as its member
    _info.shape(shape);
    return _tensor->applyShape(shape);
  }

private:
  const ir::OperandInfo _orig_info;
  const ir::Layout _orig_layout;
  bool _is_dynamic{false};
  IPortableTensor *_tensor{nullptr};        //< The actual tensor that is indirected
  std::unique_ptr<UserTensor> _user_tensor; //< If it is a user tensor, it is managed by this object
};

} // namespace controlflow
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CONTROLFLOW_INDIRECT_TENSOR_H__
