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

#ifndef __ONERT_BACKEND_ITENSOR_REGISTRY__
#define __ONERT_BACKEND_ITENSOR_REGISTRY__

#include <memory>

#include "ir/Index.h"
#include "backend/ITensor.h"

namespace onert
{
namespace backend
{

struct ITensorRegistry
{
  /**
   * @brief Deconstruct itself
   */
  virtual ~ITensorRegistry() = default;

  /**
   * @brief Returns pointer of ITensor among native and migrant tensors
   *
   * Native Tensor is a tensor that is managed by this backend
   * Migrant Tensor is a tensor that is imported from another backend
   *
   * @note  Return tensor cannot be used longer than dynamic tensor manager
   */
  virtual std::shared_ptr<ITensor> getITensor(const ir::OperandIndex &) = 0;
  /**
   * @brief Returns pointer of ITensor among native tensors
   *
   * Unlike @c getITensor , this function only searches from native tensors
   *
   * @note  Returned tensor cannot be used longer than dynamic tensor manager
   */
  virtual std::shared_ptr<ITensor> getNativeITensor(const ir::OperandIndex &) = 0;
};

} // namespace backend
} // namespace onert

#include "ir/OperandIndexMap.h"
#include "backend/IPortableTensor.h"

namespace onert
{
namespace backend
{

/**
 * @brief  TensorRegistry template class for the convenience of backend implementations
 *
 * If a backend uses @c IPortableTensor , and there is no special reason to implement @c
 * ITensorRegistry on your own, you may just use this default implementation.
 *
 * @tparam T_Tensor Tensor type. Must be a subclass of @c onert::backend::IPortableTensor .
 */
template <typename T_Tensor> class PortableTensorRegistryTemplate : public ITensorRegistry
{
public:
  std::shared_ptr<ITensor> getITensor(const ir::OperandIndex &ind) override
  {
    static_assert(std::is_base_of<ITensor, T_Tensor>::value, "T_Tensor must derive from ITensor.");
    auto external_tensor = _external.find(ind);
    if (external_tensor != _external.end())
      return external_tensor->second;
    return getNativeTensor(ind);
  }

  std::shared_ptr<ITensor> getNativeITensor(const ir::OperandIndex &ind) override
  {
    return getNativeTensor(ind);
  }

  std::shared_ptr<IPortableTensor> getPortableTensor(const ir::OperandIndex &ind)
  {
    auto external_tensor = _external.find(ind);
    if (external_tensor != _external.end())
    {
      if (external_tensor->second)
        return external_tensor->second;
    }
    return getNativeTensor(ind);
  }

  std::shared_ptr<T_Tensor> getNativeTensor(const ir::OperandIndex &ind)
  {
    auto tensor = _managed.find(ind);
    if (tensor != _managed.end())
      return tensor->second;
    return nullptr;
  }

  bool setMigrantTensor(const ir::OperandIndex &ind, const std::shared_ptr<IPortableTensor> &tensor)
  {
    // TODO Uncomment this as two tensors for an index is not allowed.
    //      But now it is temporarily allowed as a workaround. External one hides Managed one.
    // auto itr = _managed.find(ind);
    // if (itr != _managed.end() && itr->second != nullptr && tensor != nullptr)
    //  throw std::runtime_error{
    //      "Tried to set an migrant tensor but an native tensor already exists."};
    _external[ind] = tensor;
    return true;
  }

  void setNativeTensor(const ir::OperandIndex &ind, const std::shared_ptr<T_Tensor> &tensor)
  {
    auto itr = _external.find(ind);
    if (itr != _external.end() && itr->second != nullptr && tensor != nullptr)
      throw std::runtime_error{
          "Tried to set a native tensor but an migrant tensor already exists."};
    _managed[ind] = tensor;
  }

  const ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &managed_tensors() { return _managed; }

  const ir::OperandIndexMap<std::shared_ptr<IPortableTensor>> &external_tensors()
  {
    return _external;
  }

private:
  ir::OperandIndexMap<std::shared_ptr<IPortableTensor>> _external;
  ir::OperandIndexMap<std::shared_ptr<T_Tensor>> _managed;
};

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ITENSOR_REGISTRY__
