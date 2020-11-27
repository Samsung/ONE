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

#ifndef __ONERT_COMPILER_TENSOR_BUILDERS_H__
#define __ONERT_COMPILER_TENSOR_BUILDERS_H__

#include <unordered_set>
#include <memory>
#include "backend/BackendContext.h"
#include "backend/Backend.h"
#include "backend/controlflow/Config.h"
#include "backend/controlflow/TensorBuilder.h"
#include "util/logging.h"

namespace onert
{
namespace compiler
{

class TensorBuilders
{
public:
  TensorBuilders() = default;

  TensorBuilders(const onert::backend::BackendContexts &backend_contexts, bool include_controlflow)
  {
    for (const auto &e : backend_contexts)
    {
      if (e.first->config()->id() == backend::controlflow::Config::ID)
      {
        _cf_tensor_builder =
          std::dynamic_pointer_cast<backend::controlflow::TensorBuilder>(e.second->tensor_builder);
        if (include_controlflow)
          _tensor_builders.insert(e.second->tensor_builder);
      }
      else
      {
        _tensor_builders.insert(e.second->tensor_builder);
      }
    }
  }

  std::unordered_set<std::shared_ptr<onert::backend::ITensorBuilder>>::const_iterator begin() const
  {
    return _tensor_builders.cbegin();
  }
  std::unordered_set<std::shared_ptr<onert::backend::ITensorBuilder>>::const_iterator end() const
  {
    return _tensor_builders.cend();
  }

  std::shared_ptr<backend::controlflow::TensorBuilder> getControlflowTensorBuilder() const
  {
    return _cf_tensor_builder;
  }

private:
  std::unordered_set<std::shared_ptr<backend::ITensorBuilder>> _tensor_builders;
  std::shared_ptr<backend::controlflow::TensorBuilder> _cf_tensor_builder;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_TENSOR_BUILDERS_H__
