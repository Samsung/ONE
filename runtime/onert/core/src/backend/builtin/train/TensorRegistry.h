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

#ifndef __ONERT_BACKEND_BUILTIN_TRAIN_TENSOR_REGISTRY_H__
#define __ONERT_BACKEND_BUILTIN_TRAIN_TENSOR_REGISTRY_H__

#include <backend/train/ITensorRegistry.h>

#include "../IOTensor.h"
#include "../Tensor.h"
#include "Tensor.h"

namespace onert
{
namespace backend
{
namespace builtin
{
namespace train
{

using BaseTensorRegistry =
  backend::train::PortableTensorRegistryTemplate<Tensor, TrainableTensor, BackPropTensor,
                                                 GradientTensor>;

class TensorRegistry : public backend::train::ITensorRegistry
{
public:
  TensorRegistry() : _base_reg{new BaseTensorRegistry} {}

  ITensor *getITensor(const ir::OperandIndex &index) override
  {
    auto base_tensor = _base_reg->getITensor(index);
    if (base_tensor)
      return base_tensor;
    return getNativeIOTensor(index);
  }

  ITensor *getNativeITensor(const ir::OperandIndex &index) override
  {
    auto base_tensor = _base_reg->getNativeITensor(index);
    if (base_tensor)
      return base_tensor;
    return getNativeIOTensor(index);
  }

  IPortableTensor *getPortableTensor(const ir::OperandIndex &index)
  {
    auto base_tensor = _base_reg->getPortableTensor(index);
    if (base_tensor)
      return base_tensor;
    return getNativeIOTensor(index);
  }

  IOTensor *getNativeIOTensor(const ir::OperandIndex &index)
  {
    auto tensor = _native_io_tensors.find(index);
    if (tensor != _native_io_tensors.end())
      return tensor->second.get();
    return nullptr;
  }

  ITensor *getBackPropITensor(const ir::OperandIndex &index) override
  {
    return _base_reg->getBackPropTensor(index);
  }

  ITensor *getGradientITensor(const ir::OperandIndex &index) override
  {
    return _base_reg->getGradientTensor(index);
  }

  BackPropTensor *getBackPropTensor(const ir::OperandIndex &index)
  {
    return _base_reg->getBackPropTensor(index);
  }

  bool setMigrantTensor(const ir::OperandIndex &index, IPortableTensor *tensor) override
  {
    assert(tensor);
    assert(!getITensor(index)); // For the index, tensor is not registered yet
    _base_reg->setMigrantTensor(index, tensor);
    return true;
  }

  void iterateTrainableTensors(
    const std::function<void(const ir::OperandIndex &, const backend::train::ITrainableTensor *)> &)
    const override
  {
    // DO NOTHING
    // Builtin tensor registry does not have trainable tensor.
  }

  void setBackPropTensor(const ir::OperandIndex &index, std::unique_ptr<BackPropTensor> tensor)
  {
    _base_reg->setBackPropTensor(index, std::move(tensor));
  }

  void setGradientTensor(const ir::OperandIndex &index, std::unique_ptr<GradientTensor> tensor)
  {
    _base_reg->setGradientTensor(index, std::move(tensor));
  }

  void setNativeIOTensor(ir::OperandIndex index, std::unique_ptr<IOTensor> &&tensor)
  {
    assert(tensor);
    assert(!getITensor(index)); // For the index, tensor is not registered yet
    _native_io_tensors[index] = std::move(tensor);
  }

  const ir::OperandIndexMap<std::unique_ptr<IOTensor>> &native_io_tensors()
  {
    return _native_io_tensors;
  }
  std::shared_ptr<BaseTensorRegistry> base_reg() { return _base_reg; }

private:
  std::shared_ptr<BaseTensorRegistry> _base_reg;
  ir::OperandIndexMap<std::unique_ptr<IOTensor>> _native_io_tensors;
};

} // namespace train
} // namespace builtin
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BUILTIN_TRAIN_TENSOR_REGISTRY_H__
