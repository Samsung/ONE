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

#ifndef __ONERT_BACKEND_I_PORTABLE_TENSOR_H__
#define __ONERT_BACKEND_I_PORTABLE_TENSOR_H__

#include "backend/ITensor.h"
#include "ir/OperandInfo.h"
#include "ir/Sparsity.h"

namespace onert::backend
{

/**
 * @brief A tensor class that is portable for other backends
 *
 * Backends that use derivatives of this interface can reuse each other's tensors without copying.
 * Here's criterion to be a portable tensor:
 *   - it must not have any paddings
 *   - No special operations on @c access method
 *     - e.g. CL memory must map/unmap to use it from CPU, the memory so it cannot be portable
 */
class IPortableTensor : public ITensor
{
public:
  IPortableTensor(const ir::OperandInfo &info) : _info(info) {}

  virtual ~IPortableTensor();

public:
  // It is introduced to reduce virtual method call overhead on inference
  // So derived class should maintain actual OperandInfo to "_info" field
  // Ex. CPU backend getShape() in OperationUtils.h is called frequently on inference
  //     and it calls get_info()
  const ir::OperandInfo &get_info() const { return _info; }
  const ir::Sparsity *sparsity() const { return _info.typeInfo().sparsity(); }

  // Finalized methods for IPortableTensor by "_info" field read
  size_t total_size() const override final { return _info.total_size(); }
  size_t calcOffset(const ir::Coordinates &coords) const override final;
  ir::DataType data_type() const override final { return _info.typeInfo().type(); }
  float data_scale() const override final { return _info.typeInfo().scale(); }
  int32_t data_zero_point() const override final { return _info.typeInfo().zero_point(); }
  const std::vector<float> &data_scales() const override final { return _info.typeInfo().scales(); }
  const std::vector<int32_t> &data_zero_points() const override
  {
    return _info.typeInfo().zero_points();
  }
  bool is_constant() const override final { return _info.isConstant(); }
  bool is_dynamic() const override final { return _info.isDynamic(); }
  ir::Shape getShape() const override final { return _info.shape(); }

  // Finalized methods for IPortableTensor by no padding
  bool has_padding() const final { return false; }
  void access(const std::function<void(ITensor &tensor)> &fn) final { fn(*this); }

protected:
  ir::OperandInfo _info;
};

} // namespace onert::backend

#endif // __ONERT_BACKEND_I_PORTABLE_TENSOR_H__
