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

#ifndef __ONERT_COMPILER_TENSOR_REGISTRIES_H__
#define __ONERT_COMPILER_TENSOR_REGISTRIES_H__

#include "../backend/builtin/Config.h"
#include "../backend/builtin/TensorRegistry.h"

#include "backend/Backend.h"
#include "backend/BackendContext.h"

#include <memory>
#include <unordered_set>

#include "backend/train/TrainableBackendContext.h"
#include <misc/polymorphic_downcast.h>

namespace onert
{
namespace compiler
{

class TensorRegistries
{
public:
  TensorRegistries() = default;

  TensorRegistries(const onert::backend::BackendContexts &backend_contexts, bool include_builtin)
  {
    for (const auto &e : backend_contexts)
    {
      auto tensor_reg = e.second->tensor_registry;
      if (e.first->config()->id() == backend::builtin::Config::ID)
      {
        _builtin_tensor_reg =
          std::dynamic_pointer_cast<backend::builtin::TensorRegistry>(tensor_reg);
        if (include_builtin)
          _tensor_regs.insert(tensor_reg);
      }
      else
      {
        _tensor_regs.insert(tensor_reg);
      }

      if (e.first->config()->supportTraining())
      {
        auto backend_ctx = e.second.get();
        auto grad_tensor_reg =
          nnfw::misc::polymorphic_downcast<backend::train::TrainableBackendContext *>(backend_ctx)
            ->grad_tensor_registry;
        _grad_tensor_regs.insert(grad_tensor_reg);
      }
    }
  }

  std::unordered_set<std::shared_ptr<onert::backend::ITensorRegistry>>::const_iterator begin() const
  {
    return _tensor_regs.cbegin();
  }
  std::unordered_set<std::shared_ptr<onert::backend::ITensorRegistry>>::const_iterator end() const
  {
    return _tensor_regs.cend();
  }

  std::shared_ptr<backend::builtin::TensorRegistry> getBuiltinTensorRegistry() const
  {
    return _builtin_tensor_reg;
  }

  backend::ITensor *getITensor(ir::OperandIndex ind) const
  {
    for (auto &&tensor_reg : _tensor_regs)
    {
      auto tensor = tensor_reg->getITensor(ind);
      if (tensor)
        return tensor;
    }
    return nullptr;
  }

private:
  std::unordered_set<std::shared_ptr<backend::ITensorRegistry>> _tensor_regs;
  std::unordered_set<std::shared_ptr<backend::ITensorRegistry>> _grad_tensor_regs;
  std::shared_ptr<backend::builtin::TensorRegistry> _builtin_tensor_reg;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TENSOR_REGISTRIES_H__
