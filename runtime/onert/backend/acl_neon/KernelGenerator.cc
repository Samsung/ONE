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

#include "KernelGenerator.h"

#include <arm_compute/runtime/NEON/NEFunctions.h>   // Include all ARM Compute NEON functions
#include <arm_compute/runtime/NEON/NEFunctionsEx.h> // Include all ARM Compute EX NEON functions

#include <AclActivationBuilder.h>
#include <AclFunction.h>
#include <Convert.h>
#include <Swizzle.h>

#include "ir/Index.h"
#include "ir/DataType.h"
#include "ir/InternalType.h"
#include "exec/NopFunction.h"
#include "util/logging.h"
#include "AclKernelGen.h"

namespace onert::backend::acl_neon
{

using ::onert::backend::acl_common::asAclFunction;
using ActivationBuilder = ::onert::backend::acl_common::AclActivationBuilder<
  ::arm_compute::ITensor, ::arm_compute::NEActivationLayer, acl_common::AclFunction>;

KernelGenerator::KernelGenerator(
  const ir::Graph &graph, const std::shared_ptr<TensorBuilder> &tensor_builder,
  const std::shared_ptr<acl_common::AclTensorRegistry<TensorManager>> &tensor_reg)
  : basic::KernelGeneratorBase{graph}, _ctx(graph.operands()), _operations_ctx(graph.operations()),
    _tensor_builder(tensor_builder), _tensor_reg(tensor_reg)
{
  // DO NOTHING
}

std::unique_ptr<exec::FunctionSequence> KernelGenerator::generate(ir::OperationIndex ind)
{
  auto ret = std::make_unique<exec::FunctionSequence>();
  ret->enableDynamicShapeInferer(false);

  const auto &op = _graph.operations().at(ind);
  op.accept(*this);
  ret->append(releaseFunction());
  return ret;
}

} // namespace onert::backend::acl_neon
