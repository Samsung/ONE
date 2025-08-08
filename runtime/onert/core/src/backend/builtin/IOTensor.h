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

namespace onert::backend::builtin
{

/**
 * @brief Tensor object that indirects to the tensor it is pointing to.
 *
 * A executor's I/O tensor could be two types.
 *
 * 1. @c UserTensor, if it is the primary graph (package's input/output)
 * 2. Any other derivative of @c IPortableTensor from another executor, otherwise
 *
 * To support these, this object indirects everything to the actual tensor pointer.
 *
 * IOTensor is derived from IPortableTensor, and it also have "_info" field.
 * "_info" field is accessed by IPortableTensor's getter method.
 *
 * It assumes that IOTensor's info is always same with actual tensor's info except shape.
 * setTensor() updates IOTensor's info's shape to actual tensor shape.
 * Actual tensor's info should not be updated directly after setTensor() call until
 * executor's execution is finished, instead it is allowed to update actual tensor's info
 * indirectly by IOTensor's setter methods.
 */
class IOTensor : public IPortableTensor
{
public:
  IOTensor(const ir::OperandInfo &info);
  ~IOTensor();

public:
  void setTensor(IPortableTensor *tensor);
  void setBackendTensor(IPortableTensor *tensor);

public:
  uint8_t *buffer() const override { return _tensor->buffer(); }
  void set_dynamic() override
  {
    _info.setDynamic();
    _tensor->set_dynamic();
  }
  void setShape(const ir::Shape &shape) override
  {
    _info.shape(shape);
    _tensor->setShape(shape);
  }

  /*
   * Changes tensor shape and allocate memory since its shape was changed
   * perhaps by nnfw_set_input_tensorinfo()
   *
   * Cases are:
   * 1) static operand -> nnfw_set_input_tensorinfo() -> execute() -> execute()
   *                                                 (a)          (b)
   *
   * at (a), operand is static, tensor is static - memory dealloc is not needed
   *   (DynamicTensorManager cannot dealloc memory allocated by StaticTensorManager)
   * at (b), operand is static, tensor is dynamic - memory dealloc is needed
   *
   * 2) dynamic operand -> nnfw_set_input_tensorinfo() -> execute() -> execute()
   *                                                  (a)          (b)
   *
   * at (a), operand is dynamic, tensor is dynamic - memory dealloc is not needed
   *                                       since it has not been allocated yet
   * at (b), operand is dynamic, tensor is dynamic - memory dealloc is needed
   */
  bool applyShape(const ir::Shape &shape) override
  {
    auto return_val = _tensor->applyShape(shape);
    if (return_val)
    {
      _info.shape(shape);
      _info.setDynamic();
    }
    return return_val;
  }

  /**
   * @brief Return whether the tensor member has backend tensor
   */
  bool hasBackendTensor() const { return _has_backend_tensor; }

  /**
   * @brief Synchronize this IOTensor's operand info from the backend tensor
   *        Copies the full OperandInfo from _tensor to _info if the backend tensor is dynamic
   */
  void syncInfoFromBackendTensor()
  {
    assert(_tensor != nullptr);
    assert(_has_backend_tensor);
    if (_tensor->is_dynamic())
    {
      _info = _tensor->get_info();
    }
  }

private:
  IPortableTensor *_tensor{nullptr}; //< The actual tensor that is indirected
  // "_orig" has UserTensor type original tensor's info with nullptr buffer,
  // and "_tensor" points to "_user_tensor".
  // After 1st setTensor(tensor) call, "_tensor" is updated to actual tensor
  std::unique_ptr<UserTensor> _orig; //< If it is a user tensor, it is managed by this object
  bool _has_backend_tensor;
};

} // namespace onert::backend::builtin

#endif // __ONERT_BACKEND_BUILTIN_IO_TENSOR_H__
