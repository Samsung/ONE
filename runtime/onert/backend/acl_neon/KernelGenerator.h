/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_ACL_NEON_KERNEL_GENERATOR_H__
#define __ONERT_BACKEND_ACL_NEON_KERNEL_GENERATOR_H__

#include <backend/basic/KernelGeneratorBase.h>

#include "ir/Operands.h"
#include "TensorBuilder.h"
#include "AclTensorRegistry.h"
#include "TensorManager.h"

namespace onert::backend::acl_neon
{

class KernelGenerator : public basic::KernelGeneratorBase
{
public:
  KernelGenerator(const ir::Graph &graph, const std::shared_ptr<TensorBuilder> &tensor_builder,
                  const std::shared_ptr<acl_common::AclTensorRegistry<TensorManager>> &_tensor_reg);

  std::unique_ptr<exec::FunctionSequence> generate(ir::OperationIndex ind) override;

private:
#define OP(InternalName) void visit(const ir::operation::InternalName &) override;
#include "Operation.lst"
#undef OP

private:
  const ir::Operands &_ctx;
  const ir::Operations &_operations_ctx;
  std::shared_ptr<TensorBuilder> _tensor_builder;
  std::shared_ptr<acl_common::AclTensorRegistry<TensorManager>> _tensor_reg;
};

} // namespace onert::backend::acl_neon

#endif // __ONERT_BACKEND_ACL_NEON_KERNEL_GENERATOR_H__
