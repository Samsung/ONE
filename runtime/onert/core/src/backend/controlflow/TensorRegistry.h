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

#include "backend/cpu_common/TensorRegistry.h"
#include "backend/ITensorRegistry.h"
#include "Tensor.h"
#include "UserTensor.h"
#include <assert.h>

namespace onert
{
namespace backend
{
namespace controlflow
{

/**
 * @brief Tensor registry class for controlflow backend
 *
 * This class contains three types of tensors. Two native tensors(tensors that are managed by this
 * backend) and the other is migrant tensor.
 *
 * - NativeUserTensor - @c UserTensor managed by this backend, buffer is user-given
 * - NativeOwnTensor  - @c cpu_common::Tensor managed by this backend ( in @c _base_reg )
 * - MigrantTensor    - @c IPortableTensor managed by other backends ( in @c _base_reg )
 *
 * @note @c _base_reg is used in implementation to reuse @c cpu_common::StaticTensorManager
 *
 */
class TensorRegistry : public ITensorRegistry
{
public:
  TensorRegistry() : _base_reg{new cpu_common::TensorRegistry} {}

  std::shared_ptr<ITensor> getITensor(const ir::OperandIndex &ind) override
  {
    auto base_tensor = _base_reg->getITensor(ind);
    if (base_tensor)
      return base_tensor;
    return getNativeUserTensor(ind);
  }

  std::shared_ptr<ITensor> getNativeITensor(const ir::OperandIndex &ind) override
  {
    auto base_tensor = _base_reg->getNativeITensor(ind);
    if (base_tensor)
      return base_tensor;
    return getNativeUserTensor(ind);
  }

  std::shared_ptr<IPortableTensor> getPortableTensor(const ir::OperandIndex &ind)
  {
    auto base_tensor = _base_reg->getPortableTensor(ind);
    if (base_tensor)
      return base_tensor;
    return getNativeUserTensor(ind);
  }

  std::shared_ptr<IPortableTensor> getNativeTensor(const ir::OperandIndex &ind)
  {
    auto base_tensor = _base_reg->getNativeTensor(ind);
    if (base_tensor)
      return base_tensor;
    return getNativeUserTensor(ind);
  }

  std::shared_ptr<Tensor> getNativeOwnTensor(const ir::OperandIndex &ind)
  {
    return _base_reg->getNativeTensor(ind);
  }

  std::shared_ptr<UserTensor> getNativeUserTensor(const ir::OperandIndex &ind)
  {
    auto tensor = _native_user_tensors.find(ind);
    if (tensor != _native_user_tensors.end())
      return tensor->second;
    return nullptr;
  }

  bool setMigrantTensor(ir::OperandIndex ind, const std::shared_ptr<IPortableTensor> &tensor)
  {
    assert(tensor);
    assert(!getITensor(ind)); // For the ind, tensor is not registered yet
    _base_reg->setMigrantTensor(ind, tensor);
    return true;
  }

  void setNativeOwnTensor(ir::OperandIndex ind, const std::shared_ptr<Tensor> &tensor)
  {
    assert(tensor);
    assert(!getITensor(ind)); // For the ind, tensor is not registered yet
    _base_reg->setNativeTensor(ind, tensor);
  }

  void setNativeUserTensor(ir::OperandIndex ind, const std::shared_ptr<UserTensor> &tensor)
  {
    assert(tensor);
    assert(!getITensor(ind)); // For the ind, tensor is not registered yet
    _native_user_tensors[ind] = tensor;
  }

  const ir::OperandIndexMap<std::shared_ptr<UserTensor>> &native_user_tensors()
  {
    return _native_user_tensors;
  }
  std::shared_ptr<cpu_common::TensorRegistry> base_reg() { return _base_reg; }

private:
  std::shared_ptr<cpu_common::TensorRegistry> _base_reg;
  ir::OperandIndexMap<std::shared_ptr<UserTensor>> _native_user_tensors;
};

} // namespace controlflow
} // namespace backend
} // namespace onert
