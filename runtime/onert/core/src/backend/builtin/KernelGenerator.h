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

#ifndef __ONERT_BACKEND_BUILTIN_KERNEL_GENERATOR_H__
#define __ONERT_BACKEND_BUILTIN_KERNEL_GENERATOR_H__

#include "DynamicTensorManager.h"
#include "ExternalContext.h"
#include "TensorRegistry.h"
#include "../../compiler/TensorRegistries.h"

#include "backend/basic/KernelGeneratorBase.h"
#include "backend/CustomKernelBuilder.h"
#include "exec/IExecutors.h"
#include "ir/Graph.h"

namespace onert::backend::builtin
{

class KernelGenerator : public basic::KernelGeneratorBase
{
public:
  KernelGenerator(const ir::Graph &graph, DynamicTensorManager *dyn_tensor_manager,
                  const std::shared_ptr<TensorRegistry> &tensor_reg,
                  const std::shared_ptr<custom::IKernelBuilder> &kernel_builder,
                  const std::shared_ptr<ExternalContext> &external_context);

  void setTensorRegistries(const compiler::TensorRegistries &tensor_registries)
  {
    _tensor_registries = tensor_registries;
  }
  void setExecutors(const std::shared_ptr<exec::IExecutors> &executors)
  {
    // FIXME Using shared_ptr's raw pointer!
    _executors = executors.get();
  }

  void setModelIndex(const ir::ModelIndex &index) { _model_index = index; }

  std::unique_ptr<exec::FunctionSequence> generate(ir::OperationIndex ind) override;

private:
  void visit(const ir::operation::Custom &) override;
  void visit(const ir::operation::Call &) override;
  void visit(const ir::operation::If &) override;
  void visit(const ir::operation::Permute &) override;
  void visit(const ir::operation::While &) override;

private:
  backend::ITensor *getTensor(const ir::OperandIndex &index);
  backend::IPortableTensor *getPortableTensor(const ir::OperandIndex &index);

private:
  DynamicTensorManager *_dyn_tensor_manager;
  std::shared_ptr<TensorRegistry> _tensor_reg;
  compiler::TensorRegistries _tensor_registries;
  exec::IExecutors *_executors;
  ir::ModelIndex _model_index;
  std::shared_ptr<backend::custom::IKernelBuilder> _kernel_builder;
  const std::shared_ptr<ExternalContext> _external_context;
};

} // namespace onert::backend::builtin

#endif // __ONERT_BACKEND_BUILTIN_KERNEL_GENERATOR_H__
