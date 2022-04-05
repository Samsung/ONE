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

#include "KernelGenerator.h"

#include <backend/Backend.h>
#include <backend/IConfig.h>
#include <memory>
#include <util/Utils.h>
#include <util/logging.h>
#include <exec/DynamicShapeInferer.h>

#include <stdexcept>

namespace onert
{
namespace backend
{
namespace trix
{

KernelGenerator::KernelGenerator(const ir::Graph &graph,
                                 const std::shared_ptr<TensorBuilder> &tensor_builder,
                                 const std::shared_ptr<basic::TensorRegistry> &tensor_reg)
  : basic::KernelGeneratorBase{graph},
    _ctx(graph.operands()), _operations_ctx{graph.operations()}, _current_layout{graph.layout()},
    _tensor_builder(tensor_builder), _tensor_reg{tensor_reg}
{
  // DO NOTHING
}

std::unique_ptr<exec::FunctionSequence> KernelGenerator::generate(ir::OperationIndex ind)
{
  auto ret = std::make_unique<exec::FunctionSequence>();

  assert(_tensor_builder->dynamicTensorManager());
  assert(_tensor_reg);

  auto dyn_shape_inferer = std::make_shared<exec::DynamicShapeInferer>(_ctx, _tensor_reg);

  // Prepare to handle dynamic tensors later
  auto dyn_ctx = std::make_shared<exec::FunctionSequence::DynamicTensorCtx>();
  {
    dyn_ctx->op_ind = ind;
    dyn_ctx->operations = &_operations_ctx;
    dyn_ctx->dynamic_shape_inferer = std::move(dyn_shape_inferer);

    ret->dynamic_tensor_ctx(dyn_ctx);
  }

  auto &op = _graph.operations().at(ind);
  op.accept(*this);
  assert(_return_fn); // _return_fn must have been generated
  ret->append(std::move(_return_fn));

  for (auto ind : (op.getInputs() | ir::Remove::UNDEFINED) + op.getOutputs())
  {
    auto portable_tensor = _tensor_reg->getPortableTensor(ind);
    if (portable_tensor)
    {
      assert(portable_tensor->layout() == ir::Layout::NHWC);
    }

    auto tensor = _tensor_reg->getNativeTensor(ind);
    if (tensor)
    {
      tensor->increase_ref();
    }
  }
  return ret;
}

} // namespace trix
} // namespace backend
} // namespace onert
