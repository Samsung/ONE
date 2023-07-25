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
#include "ops/ElementwiseActivationLayer.h"
#include "ops/GradientApplier.h"
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
ops::ElementwiseActivationType
convertElementwiseActivationType(ir::operation::ElementwiseActivation::Type type_ir)
{
  switch (type_ir)
  {
    case ir::operation::ElementwiseActivation::Type::RELU:
      return ops::ElementwiseActivationType::kReLU;
    default:
      throw std::runtime_error("train KernelGenerator : Not supported operation yet");
  }
}

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
  auto ret = std::make_unique<exec::train::TrainableFnSequence>();

  const auto &op = _tgraph.operation(idx);
  op.accept(*this);
  assert(_return_fn);
  ret->append(std::move(_return_fn));

  if (_update_fn)
    ret->append(std::move(_update_fn));

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
                                 const std::shared_ptr<ExternalContext> &external_context,
                                 std::shared_ptr<exec::train::optimizer::Optimizer> optimizer)
  : backend::train::KernelGeneratorBase{tgraph}, _current_layout{tgraph.layout()},
    _tensor_reg{tensor_reg},
    _external_context(external_context), _optimizer{optimizer}, _update_fn{nullptr}
{
  // DO NOTHING
}

void KernelGenerator::visit(const ir::train::operation::Conv2D &node)
{
  // TODO Generate kernel

  // Generate GradientApplier
  const auto ker_index{node.getInputs().at(ir::train::operation::Conv2D::Input::KERNEL)};

  auto grad_tensor = _tensor_reg->getGradientTensor(ker_index);
  auto ker_tensor = _tensor_reg->getTrainableTensor(ker_index);

  auto update_fn = std::make_unique<ops::GradientApplier>();

  update_fn->configure(_optimizer, grad_tensor, ker_tensor);

  _update_fn = std::move(update_fn);
}

void KernelGenerator::visit(const ir::train::operation::ElementwiseActivation &node)
{
  using ir::train::operation::ElementwiseActivation;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ElementwiseActivation::Input::INPUT)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto deriv_input_tensor = _tensor_reg->getDerivativeTensor(input_index);
  auto deriv_output_tensor = _tensor_reg->getDerivativeTensor(output_index);

  auto fn = std::make_unique<ops::ElementwiseActivationLayer>();

  fn->configure(input_tensor, output_tensor, deriv_input_tensor, deriv_output_tensor,
                node.param().alpha, node.param().beta,
                convertElementwiseActivationType(node.param().op_type));

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::train::operation::Loss &)
{
  // TODO Generate kernel
  UNUSED_RELEASE(convertPoolType);
}

} // namespace train
} // namespace backend
} // namespace onert
