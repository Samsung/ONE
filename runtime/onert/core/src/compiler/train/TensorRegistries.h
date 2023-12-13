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

#ifndef __ONERT_COMPILER_TRAIN_TENSOR_REGISTRIES_H__
#define __ONERT_COMPILER_TRAIN_TENSOR_REGISTRIES_H__

#include "../../backend/builtin/Config.h"
#include "../../backend/builtin/train/TensorRegistry.h"

#include <backend/train/ITensorRegistry.h>
#include <backend/train/ITrainableTensor.h>
#include <backend/train/TrainableBackendContext.h>

#include <memory>
#include <unordered_set>

namespace onert
{
namespace compiler
{
namespace train
{

class TensorRegistries
{
public:
  TensorRegistries() = default;

  TensorRegistries(const backend::train::TrainableBackendContexts &backend_contexts,
                   bool include_builtin)
  {
    for (const auto &e : backend_contexts)
    {
      auto tensor_reg = e.second->tensor_registry();
      if (e.first->config()->id() == backend::builtin::Config::ID)
      {
        _builtin_tensor_reg =
          std::dynamic_pointer_cast<backend::builtin::train::TensorRegistry>(tensor_reg);
        if (include_builtin)
          _tensor_regs.insert(tensor_reg);
      }
      else
      {
        _tensor_regs.insert(tensor_reg);
      }
    }
  }

  std::unordered_set<std::shared_ptr<backend::train::ITensorRegistry>>::const_iterator begin() const
  {
    return _tensor_regs.cbegin();
  }
  std::unordered_set<std::shared_ptr<backend::train::ITensorRegistry>>::const_iterator end() const
  {
    return _tensor_regs.cend();
  }

  std::shared_ptr<backend::builtin::train::TensorRegistry> getBuiltinTensorRegistry() const
  {
    return _builtin_tensor_reg;
  }

  backend::ITensor *getITensor(ir::OperandIndex index) const
  {
    for (auto &&tensor_reg : _tensor_regs)
    {
      auto tensor = tensor_reg->getITensor(index);
      if (tensor)
        return tensor;
    }
    return nullptr;
  }

  backend::ITensor *getBackPropITensor(ir::OperandIndex index) const
  {
    for (auto &&tensor_reg : _tensor_regs)
    {
      auto tensor = tensor_reg->getBackPropITensor(index);
      if (tensor)
        return tensor;
    }
    return nullptr;
  }

  void iterateTrainableTensors(
    const std::function<void(const ir::OperandIndex &, const backend::train::ITrainableTensor *)>
      &fn) const
  {
    assert(_tensor_regs.size() == 2); // training backend and built-in backend
    for (const auto &tensor_reg : _tensor_regs)
      tensor_reg->iterateTrainableTensors(fn);
  }

private:
  std::unordered_set<std::shared_ptr<backend::train::ITensorRegistry>> _tensor_regs;
  std::shared_ptr<backend::builtin::train::TensorRegistry> _builtin_tensor_reg;
};

} // namespace train
} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TRAIN_TENSOR_REGISTRIES_H__
