/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRIX_KERNEL_GENERATOR_H__
#define __ONERT_BACKEND_TRIX_KERNEL_GENERATOR_H__

#include "TensorBuilder.h"
#include "backend/basic/TensorRegistry.h"
#include "Tensor.h"
#include "DevContext.h"

#include <backend/basic/KernelGeneratorBase.h>
#include <ir/Operands.h>
#include <ir/Operations.h>

namespace onert
{
namespace backend
{
namespace trix
{

class KernelGenerator : public basic::KernelGeneratorBase
{
public:
  KernelGenerator(const ir::Graph &graph, const std::shared_ptr<TensorBuilder> &tensor_builder,
                  const std::shared_ptr<basic::TensorRegistry> &tensor_reg,
                  const std::shared_ptr<DevContext> &dev_context);

  std::unique_ptr<exec::FunctionSequence> generate(ir::OperationIndex op_ind) override;

private:
  void visit(const ir::operation::Bulk &node) override;

private:
  const ir::Operands &_ctx;
  const ir::Operations &_operations_ctx;
  std::shared_ptr<TensorBuilder> _tensor_builder;
  std::shared_ptr<basic::TensorRegistry> _tensor_reg;
  const std::shared_ptr<DevContext> _dev_context;
};

} // namespace trix
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRIX_KERNEL_GENERATOR_H__
