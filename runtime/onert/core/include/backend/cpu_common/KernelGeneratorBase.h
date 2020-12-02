/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_CPU_COMMON_KERNEL_GENERATOR_BASE_H__
#define __ONERT_BACKEND_CPU_COMMON_KERNEL_GENERATOR_BASE_H__

#include <assert.h>
#include <memory>
#include <functional>

#include "ir/OperationVisitor.h"
#include "ir/OpSequence.h"
#include <memory>
#include "exec/FunctionSequence.h"
#include "backend/ITensorRegistry.h"

namespace onert
{
namespace backend
{
namespace cpu_common
{

class KernelGeneratorBase : public ir::OperationVisitor
{
public:
  virtual ~KernelGeneratorBase() = default;

  std::unique_ptr<exec::IFunction> releaseFunction()
  {
    assert(_return_fn);
    return std::move(_return_fn);
  }

  std::unique_ptr<exec::FunctionSequence> generate(const ir::OpSequence &op_seq)
  {
    op_seq.accept(*this);
    return std::move(_return_fn_seq);
  }

protected:
  using OperationVisitor::visit;

  void visit(const ir::OpSequence &) override
  {
    throw std::runtime_error("KernelGenerator: NYI for operation 'OpSequence'");
  }

#define OP(InternalName)                                                                \
  void visit(const ir::operation::InternalName &) override                              \
  {                                                                                     \
    throw std::runtime_error("KernelGenerator: NYI for operation '" #InternalName "'"); \
  }
#include "ir/Operations.lst"
#undef OP

protected:
  std::unique_ptr<exec::IFunction> _return_fn;
  std::unique_ptr<exec::FunctionSequence> _return_fn_seq; // TODO Extract this out
};

} // namespace cpu_common
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_CPU_COMMON_KERNEL_GENERATOR_BASE_H__
