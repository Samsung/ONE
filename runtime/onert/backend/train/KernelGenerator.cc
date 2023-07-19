/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ops/ConvolutionLayer.h"
#include "ops/PoolLayer.h"

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
namespace train
{

namespace
{
ops::PoolType convertPoolType(ir::operation::Pool2D::PoolType type_ir)
{
  switch (type_ir)
  {
    // TODO Implement AVG PoolType
    case ir::operation::Pool2D::PoolType::MAX:
      return ops::PoolType::kMax;
    default:
      throw std::runtime_error("train KernelGenerator : Not supported operation yet");
  }
}
} // namespace

std::unique_ptr<exec::train::TrainableFnSequence> KernelGenerator::generate(ir::OperationIndex idx)
{
  // TODO Generate TrainableFnSequence that can go backward as well
  auto ret = std::make_unique<exec::train::TrainableFnSequence>();

  const auto &op = _tgraph.operation(idx);
  op.accept(*this);
  assert(_return_fn);
  ret->append(std::move(_return_fn));

  for (auto &&ind : (op.getInputs() | ir::Remove::UNDEFINED) + op.getOutputs())
  {
    auto portable_tensor = _tensor_reg->getPortableTensor(ind);
    if (portable_tensor)
    {
      assert(portable_tensor->layout() == ir::Layout::NHWC);
    }
    auto tensor = _tensor_reg->getNonConstTensor(ind);
    if (tensor)
    {
      tensor->increase_ref();
    }
  }
  return ret;
}

KernelGenerator::KernelGenerator(const ir::train::TrainableGraph &tgraph,
                                 const std::shared_ptr<TensorRegistry> &tensor_reg,
                                 const std::shared_ptr<ExternalContext> &external_context)
  : backend::train::KernelGeneratorBase{tgraph}, _current_layout{tgraph.layout()},
    _tensor_reg{tensor_reg}, _external_context(external_context)
{
  // DO NOTHING
}

void KernelGenerator::visit(const ir::train::operation::Loss &)
{
  // TODO Generate kernel
  UNUSED_RELEASE(convertPoolType);
}

} // namespace train
} // namespace backend
} // namespace onert
