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

#ifndef __ONERT_BACKEND_CONTROLFLOW_KERNEL_GENERATOR_H__
#define __ONERT_BACKEND_CONTROLFLOW_KERNEL_GENERATOR_H__

#include <backend/IKernelGenerator.h>
#include <backend/ITensorBuilder.h>
#include <exec/IExecutor.h>
#include <ir/Operands.h>
#include <ir/Operations.Include.h>

namespace onert
{
namespace backend
{
namespace controlflow
{

class KernelGenerator : public IKernelGenerator
{
public:
  KernelGenerator(const ir::Operands &operand_ctx);

  void setTensorBuilderSet(const TensorBuilderSet &tensor_builder_set)
  {
    _tensor_builder_set = tensor_builder_set;
  }
  void setExecutorMap(const std::shared_ptr<exec::ExecutorMap> &executor_map)
  {
    _executor_map = executor_map;
  }

  using IKernelGenerator::visit;

  void visit(const ir::OpSequence &) override;
  void visit(const ir::operation::While &) override;

private:
  const ir::Operands &_operand_ctx;
  TensorBuilderSet _tensor_builder_set;
  std::shared_ptr<exec::ExecutorMap> _executor_map;
};

} // namespace controlflow
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CONTROLFLOW_KERNEL_GENERATOR_H__
