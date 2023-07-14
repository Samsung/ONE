/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_ITENSOR_REGISTRY_H__
#define __ONERT_BACKEND_TRAIN_ITENSOR_REGISTRY_H__

#include "backend/ITensorRegistry.h"

namespace onert
{
namespace backend
{
namespace train
{

class ITensorRegistry : public backend::ITensorRegistry
{
public:
  /**
   * @brief Returns pointer of ITensor among native and migrant tensors, not derivative and gradient
   *
   */
  using backend::ITensorRegistry::getITensor;

  /**
   * @brief Returns pointer of ITensor among native tensors, not derivative and gradient
   *
   */
  using backend::ITensorRegistry::getNativeITensor;

  /**
   * @brief Returns pointer of ITensor for derivative
   *
   * @note  Return tensor cannot be used longer than dynamic tensor manager
   */
  virtual ITensor *getDerivativeITensor(const ir::OperandIndex &) = 0;

  /**
   * @brief Returns pointer of ITensor for gradient
   *
   * @note  Returned tensor cannot be used longer than dynamic tensor manager
   */
  virtual ITensor *getGradientITensor(const ir::OperandIndex &) = 0;
};

} // namespace train
} // namespace backend
} // namespace onert

namespace onert
{
namespace backend
{
namespace train
{

template <typename Tensor, typename TrainableTensor, typename DerivativeTensor,
          typename GradientTensor>
class PortableTensorRegistryTemplate : public backend::train::ITensorRegistry
{
public:
  using TrainingTensors = std::tuple<TrainableTensor *, GradientTensor *>;

public:
  ITensor *getITensor(const ir::OperandIndex &index) override
  {
    auto _migrant_tensor = _migrant.find(index);
    if (_migrant_tensor != _migrant.end())
      return _migrant_tensor->second;
    return getNativeITensor(index);
  }

  ITensor *getNativeITensor(const ir::OperandIndex &index) override
  {
    ITensor *tensor = getTrainableTensor(index);
    if (tensor == nullptr)
      tensor = getNonConstTensor(index);
    return tensor;
  }

  ITensor *getDerivativeITensor(const ir::OperandIndex &index) override
  {
    return getDerivativeTensor(index);
  }

  ITensor *getGradientITensor(const ir::OperandIndex &index) override
  {
    return getGradientTensor(index);
  }

  IPortableTensor *getPortableTensor(const ir::OperandIndex &index)
  {
    auto tensor = _trainable.find(index);
    if (tensor != _trainable.end())
    {
      if (tensor->second)
        return tensor->second.get();
    }
    return getNonConstTensor(index);
  }

  Tensor *getNonConstTensor(const ir::OperandIndex &index)
  {
    auto tensor = _non_const.find(index);
    if (tensor != _non_const.end())
      return tensor->second.get();
    return nullptr;
  }

  TrainableTensor *getTrainableTensor(const ir::OperandIndex &index)
  {
    auto tensor = _trainable.find(index);
    if (tensor != _trainable.end())
      return tensor->second.get();

    return nullptr;
  }

  DerivativeTensor *getDerivativeTensor(const ir::OperandIndex &index)
  {
    auto tensor = _derivative.find(index);
    if (tensor != _derivative.end())
      return tensor->second.get();
    return nullptr;
  }

  GradientTensor *getGradientTensor(const ir::OperandIndex &index)
  {
    auto tensor = _gradient.find(index);
    if (tensor != _gradient.end())
      return tensor->second.get();
    return nullptr;
  }

  TrainingTensors getTrainingTensors(const ir::OperandIndex &index)
  {
    auto trainable = getTrainableTensor(index);
    if (trainable == nullptr)
      throw std::runtime_error{
        "Tried to get a trainable tensor but the corresponding tensor does not exist."};

    auto gradient = getGradientTensor(index);
    if (gradient == nullptr)
      throw std::runtime_error{
        "Tried to get a gradient tensor but the corresponding tensor does not exist."};

    return TrainingTensors{std::make_pair(trainable, gradient)};
  }

  bool setMigrantTensor(const ir::OperandIndex &index, IPortableTensor *tensor) override
  {
    assert(tensor != nullptr);
    if (getITensor(index) != nullptr)
      throw std::runtime_error{
        "Tried to set a trainable tensor but another tensor already exists."};

    _migrant[index] = tensor;
    return true;
  }

  void setNonConstTensor(const ir::OperandIndex &index, std::unique_ptr<Tensor> tensor)
  {
    assert(tensor != nullptr);
    if (getITensor(index) != nullptr)
      throw std::runtime_error{
        "Tried to set a trainable tensor but another tensor already exists."};

    _non_const[index] = std::move(tensor);
  }

  void setTrainableTensor(const ir::OperandIndex &index, std::unique_ptr<TrainableTensor> tensor)
  {
    assert(tensor != nullptr);
    if (getITensor(index) != nullptr)
      throw std::runtime_error{
        "Tried to set a trainable tensor but another tensor already exists."};

    _trainable[index] = std::move(tensor);
  }

  void setDerivativeTensor(const ir::OperandIndex &index, std::unique_ptr<DerivativeTensor> tensor)
  {
    assert(tensor != nullptr);
    auto itr = _derivative.find(index);
    if (itr != _derivative.end())
      throw std::runtime_error{
        "Tried to set a derivative tensor but another derivative tensor already exists."};

    _derivative[index] = std::move(tensor);
  }

  void setGradientTensor(const ir::OperandIndex &index, std::unique_ptr<GradientTensor> tensor)
  {
    assert(tensor != nullptr);
    auto itr = _gradient.find(index);
    if (itr != _gradient.end())
      throw std::runtime_error{
        "Tried to set a gradient tensor but another gradient tensor already exists."};

    _gradient[index] = std::move(tensor);
  }

  const ir::OperandIndexMap<std::unique_ptr<TrainableTensor>> &trainable_tensors()
  {
    return _trainable;
  }
  const ir::OperandIndexMap<std::unique_ptr<Tensor>> &nonconst_tensors() { return _non_const; }
  const ir::OperandIndexMap<std::unique_ptr<Tensor>> &derivative_tensors() { return _derivative; }
  const ir::OperandIndexMap<std::unique_ptr<GradientTensor>> &gradient_tensors()
  {
    return _gradient;
  }

private:
  // Native tensors
  ir::OperandIndexMap<std::unique_ptr<Tensor>> _non_const;
  ir::OperandIndexMap<std::unique_ptr<TrainableTensor>> _trainable;

  // Migrant tensors
  ir::OperandIndexMap<IPortableTensor *> _migrant;

  // Tensors for backpropagation
  ir::OperandIndexMap<std::unique_ptr<DerivativeTensor>> _derivative;

  // Tensors for updating trainable tensors
  ir::OperandIndexMap<std::unique_ptr<GradientTensor>> _gradient;
};

} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_ITENSOR_REGISTRY_H__
