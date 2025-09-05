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

#ifndef __ONERT_BACKEND_BUILTIN_USER_TENSOR_H__
#define __ONERT_BACKEND_BUILTIN_USER_TENSOR_H__

#include "backend/IPortableTensor.h"
#include "ir/OperandInfo.h"

namespace onert::backend::builtin
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
  UserTensor(const ir::OperandInfo &info, uint8_t *buffer, size_t size)
    : IPortableTensor{info}, _buffer{buffer}, _size{size}
  {
  }

public:
  uint8_t *buffer() const override { return _buffer; }
  void set_dynamic() override { _info.setDynamic(); }
  void setShape(const ir::Shape &new_shape) override { _info.shape(new_shape); }
  bool applyShape(const ir::Shape &) override;

private:
  uint8_t *_buffer;
  size_t _size;
};

} // namespace onert::backend::builtin

#endif // __ONERT_BACKEND_BUILTIN_USER_TENSOR_H__
