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

#ifndef __ONERT_BACKEND_BUILTIN_IO_TENSOR_H__
#define __ONERT_BACKEND_BUILTIN_IO_TENSOR_H__

#include "backend/IPortableTensor.h"
#include "UserTensor.h"

namespace onert
{
namespace backend
{
namespace builtin
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
  ~IOTensor();

public:
  void setTensor(IPortableTensor *tensor);
  const ir::OperandInfo &orig_info() const { return _orig->get_info(); }
  ir::Layout orig_layout() const { return _orig->layout(); }

public:
  // Some methods can be called before execution start: on compile phase
  // After compilation and before actual I/O tensor assignment by setTensor(tensor), we should use
  // above orig_xxx() methods
  const ir::OperandInfo &get_info() const override { return _tensor->get_info(); }
  float data_scale() const override { return _tensor->data_scale(); }
  int32_t data_zero_point() const override { return _tensor->data_zero_point(); }
  const std::vector<float> &data_scales() const override { return _tensor->data_scales(); }
  const std::vector<int32_t> &data_zero_points() const override
  {
    return _tensor->data_zero_points();
  }
  uint8_t *buffer() const override { return _tensor->buffer(); }
  size_t total_size() const override { return _tensor->total_size(); }
  size_t calcOffset(const ir::Coordinates &coords) const override
  {
    return _tensor->calcOffset(coords);
  }
  ir::Layout layout() const override { return _tensor->layout(); }
  ir::DataType data_type() const override { return _tensor->data_type(); }
  bool is_dynamic() const override
  {
    return _is_dynamic || _orig->is_dynamic() || _tensor->is_dynamic();
  }
  void set_dynamic() override { _tensor->set_dynamic(); }
  ir::Shape getShape() const override { return _tensor->getShape(); }
  void setShape(const ir::Shape &shape) override
  {
    _tensor->setShape(shape);
    _orig->setShape(shape);
  }
  bool is_constant() const override { return _tensor->is_constant(); }
  bool applyShape(const ir::Shape &shape) override { return _tensor->applyShape(shape); }

private:
  // IPortableTensor's info is not used
  bool _is_dynamic{false};           // < Represent dynamic by updated model input shape
  IPortableTensor *_tensor{nullptr}; //< The actual tensor that is indirected
  // Before 1st inference, "_orig" has original tensor's info with nullptr buffer
  // After 1st setTensor(tensor) call, "_orig" has latest shape info with nullptr buffer
  // We can use IPortableTensor's info for tensor info cache, but we use nullptr
  // UserTensor for simple method implementation
  std::unique_ptr<UserTensor> _orig;
};

} // namespace builtin
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BUILTIN_IO_TENSOR_H__
