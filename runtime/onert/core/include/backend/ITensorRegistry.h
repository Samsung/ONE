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
   * @brief Returns pointer of ITensor
   * @note  Return tensor cannot be used longer than dynamic tensor manager
   */
  virtual std::shared_ptr<ITensor> getITensor(const ir::OperandIndex &) = 0;
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
    return getManagedTensor(ind);
  }

  std::shared_ptr<IPortableTensor> getPortableTensor(const ir::OperandIndex &ind)
  {
    auto external_tensor = _external.find(ind);
    if (external_tensor != _external.end())
    {
      auto external_portable_tensor =
          std::dynamic_pointer_cast<IPortableTensor>(external_tensor->second);
      if (external_portable_tensor)
        return external_tensor->second;
    }
    return std::dynamic_pointer_cast<IPortableTensor>(getManagedTensor(ind));
  }

  std::shared_ptr<T_Tensor> getManagedTensor(const ir::OperandIndex &ind)
  {
    auto tensor = _managed.find(ind);
    if (tensor != _managed.end())
      return tensor->second;
    return nullptr;
  }

  void setExternalTensor(const ir::OperandIndex &ind,
                         const std::shared_ptr<IPortableTensor> &tensor)
  {
    auto itr = _managed.find(ind);
    if (itr != _managed.end() && itr->second != nullptr && tensor != nullptr)
      throw std::runtime_error{
          "Tried to set an external tensor but an managed tensor already exists."};
    _external[ind] = tensor;
  }

  void setManagedTensor(const ir::OperandIndex &ind, const std::shared_ptr<T_Tensor> &tensor)
  {
    auto itr = _external.find(ind);
    if (itr != _external.end() && itr->second != nullptr && tensor != nullptr)
      throw std::runtime_error{
          "Tried to set a managed tensor but an external tensor already exists."};
    _managed[ind] = tensor;
  }

  const ir::OperandIndexMap<std::shared_ptr<T_Tensor>> &managed_tensors() { return _managed; }

private:
  ir::OperandIndexMap<std::shared_ptr<IPortableTensor>> _external;
  ir::OperandIndexMap<std::shared_ptr<T_Tensor>> _managed;
};

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_ITENSOR_REGISTRY__
