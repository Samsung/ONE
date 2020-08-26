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

#include <unordered_set>
#include <memory>
#include "backend/BackendContext.h"
#include "backend/Backend.h"
#include "backend/controlflow/Config.h"
#include "backend/controlflow/TensorBuilder.h"
#include "backend/controlflow/TensorRegistry.h"

namespace onert
{
namespace compiler
{

class TensorRegistries
{
public:
  TensorRegistries() = default;

  TensorRegistries(const onert::backend::BackendContexts &backend_contexts,
                   bool include_controlflow)
  {
    for (const auto &e : backend_contexts)
    {
      auto tensor_reg = e.second->tensor_builder->tensorRegistry();
      if (e.first->config()->id() == backend::controlflow::Config::ID)
      {
        _cf_tensor_reg =
            std::dynamic_pointer_cast<backend::controlflow::TensorRegistry>(tensor_reg);
        if (include_controlflow)
          _tensor_regs.insert(tensor_reg);
      }
      else
      {
        _tensor_regs.insert(tensor_reg);
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

  std::shared_ptr<backend::controlflow::TensorRegistry> getControlflowTensorRegistry() const
  {
    return _cf_tensor_reg;
  }

  std::shared_ptr<backend::ITensor> getITensor(ir::OperandIndex ind)
  {
    for (auto &tensor_reg : _tensor_regs)
    {
      auto tensor = tensor_reg->getITensor(ind);
      if (tensor)
        return tensor;
    }
    return nullptr;
  }

private:
  std::unordered_set<std::shared_ptr<backend::ITensorRegistry>> _tensor_regs;
  std::shared_ptr<backend::controlflow::TensorRegistry> _cf_tensor_reg;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TENSOR_REGISTRIES_H__
