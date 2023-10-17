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
#include "ops/FullyConnectedLayer.h"
#include "ops/LossLayer.h"
#include "ops/GradientApplier.h"
#include "ops/PoolLayer.h"
#include "ops/ReshapeLayer.h"
#include "ops/SoftMaxLayer.h"

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

ops::LossType convertLossType(ir::operation::Loss::Type type_ir)
{
  switch (type_ir)
  {
    case ir::operation::Loss::Type::MEAN_SQUARED_ERROR:
      return ops::LossType::kMSE;
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

std::unique_ptr<ops::GradientApplier>
generateGradientApplier(const exec::train::optimizer::Optimizer *optimizer,
                        const IPortableTensor *gradient, ITrainableTensor *trainable)
{
  auto update_fn = std::make_unique<ops::GradientApplier>();
  update_fn->configure(optimizer, gradient, trainable);
  return update_fn;
}
} // namespace

std::unique_ptr<exec::train::TrainableFnSequence> KernelGenerator::generate(ir::OperationIndex idx)
{
  auto ret = std::make_unique<exec::train::TrainableFnSequence>();

  const auto &op = _tgraph.operation(idx);
  op.accept(*this);
  assert(_return_fn);
  ret->append(std::move(_return_fn));

  for (auto &&update_fn : _update_funcs)
    ret->append(std::move(update_fn));
  _update_funcs.clear();

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
                                 const exec::train::optimizer::Optimizer *optimizer)
  : backend::train::KernelGeneratorBase{tgraph}, _current_layout{tgraph.layout()},
    _tensor_reg{tensor_reg},
    _external_context(external_context), _optimizer{optimizer}, _update_funcs{}
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

  _update_funcs.emplace_back(generateGradientApplier(_optimizer, grad_tensor, ker_tensor));
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

void KernelGenerator::visit(const ir::train::operation::FullyConnected &node)
{
  using ir::train::operation::FullyConnected;

  const auto out_index{node.getOutputs().at(0)};
  const auto in_index{node.getInputs().at(FullyConnected::Input::INPUT)};
  const auto weights_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};
  const auto bias_index{node.getInputs().at(FullyConnected::Input::BIAS)};

  auto out_tensor = _tensor_reg->getPortableTensor(out_index);
  auto in_tensor = _tensor_reg->getPortableTensor(in_index);
  auto weights_tensor = _tensor_reg->getTrainableTensor(weights_index);
  auto bias_tensor = _tensor_reg->getTrainableTensor(bias_index);

  auto out_deriv_tensor = _tensor_reg->getDerivativeTensor(out_index);
  auto in_deriv_tensor = _tensor_reg->getDerivativeTensor(in_index);
  auto weights_grad_tensor = _tensor_reg->getGradientTensor(weights_index);
  auto bias_grad_tensor = _tensor_reg->getGradientTensor(bias_index);

  // Generate kernel
  const auto activation = node.param().activation;
  const auto weights_format = node.param().weights_format;

  auto fn = std::make_unique<ops::FullyConnectedLayer>();

  fn->configure(in_tensor, weights_tensor, bias_tensor, out_tensor, in_deriv_tensor,
                weights_grad_tensor, bias_grad_tensor, out_deriv_tensor, activation, weights_format,
                _external_context);

  _return_fn = std::move(fn);

  // Generate GradientAppliers
  if (bias_tensor)
    _update_funcs.emplace_back(generateGradientApplier(_optimizer, bias_grad_tensor, bias_tensor));
  _update_funcs.emplace_back(
    generateGradientApplier(_optimizer, weights_grad_tensor, weights_tensor));
}

void KernelGenerator::visit(const ir::train::operation::Loss &node)
{
  using ir::train::operation::Loss;

  const auto output_index{node.getOutputs().at(0)};
  const auto y_pred_index{node.getInputs().at(Loss::Y_PRED)};
  const auto y_true_index{node.getInputs().at(Loss::Y_TRUE)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto y_pred_tensor = _tensor_reg->getPortableTensor(y_pred_index);
  auto y_true_tensor = _tensor_reg->getPortableTensor(y_true_index);

  auto deriv_y_pred_tensor = _tensor_reg->getDerivativeTensor(y_pred_index);
  auto fn = std::make_unique<ops::LossLayer>();

  fn->configure(y_pred_tensor, y_true_tensor, output_tensor, deriv_y_pred_tensor,
                convertLossType(node.param().op_type));

  _return_fn = std::move(fn);

  UNUSED_RELEASE(convertPoolType);
}

void KernelGenerator::visit(const ir::train::operation::Pool2D &node)
{
  using ir::train::operation::Pool2D;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(0)};

  const auto operands = _tgraph.operands();
  const auto ofm_shape = operands.at(output_index).shape();
  const auto ifm_shape = operands.at(input_index).shape();

  if (ifm_shape.rank() != 4)
  {
    std::runtime_error(node.name() + " only supports 4D tensor as input");
  }

  // calcualate padding
  const auto stride = node.param().stride;
  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto padding =
    ir::calculatePadding(node.param().padding, ifm_shape.asFeature(_current_layout),
                         ofm_shape.asFeature(_current_layout), stride, kw, kh);

  auto out_tensor = _tensor_reg->getPortableTensor(output_index);
  auto in_tensor = _tensor_reg->getPortableTensor(input_index);

  auto out_deriv_tensor = _tensor_reg->getDerivativeTensor(output_index);
  auto in_deriv_tensor = _tensor_reg->getDerivativeTensor(input_index);

  const auto activation = node.param().activation;
  const auto pool_type = convertPoolType(node.param().op_type);

  auto fn = std::make_unique<ops::PoolLayer>();

  fn->configure(in_tensor, padding.left, padding.right, padding.top, padding.bottom,
                stride.horizontal, stride.vertical, kw, kh, activation, out_tensor, pool_type,
                in_deriv_tensor, out_deriv_tensor);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::train::operation::Reshape &node)
{
  using ir::train::operation::Reshape;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reshape::Input::INPUT)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto output_deriv_tensor = _tensor_reg->getDerivativeTensor(output_index);
  auto input_deriv_tensor = _tensor_reg->getDerivativeTensor(input_index);

  // optional 2nd input
  IPortableTensor *shape_tensor = nullptr;

  if (node.getInputs().size() == 2)
  {
    const auto shape_index{node.getInputs().at(ir::operation::Reshape::Input::SHAPE)};
    shape_tensor = _tensor_reg->getPortableTensor(shape_index);
  }

  auto fn = std::make_unique<ops::ReshapeLayer>();

  fn->configure(input_tensor, shape_tensor, output_tensor, input_deriv_tensor, output_deriv_tensor);
  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::train::operation::Softmax &node)
{
  using ir::train::operation::Softmax;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Softmax::Input::INPUT)};

  const auto beta = node.param().beta;

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);

  auto output_deriv_tensor = _tensor_reg->getDerivativeTensor(output_index);
  auto input_deriv_tensor = _tensor_reg->getDerivativeTensor(input_index);

  auto fn = std::make_unique<ops::SoftMaxLayer>();

  fn->configure(input_tensor, beta, output_tensor, input_deriv_tensor, output_deriv_tensor);
  _return_fn = std::move(fn);
}

} // namespace train
} // namespace backend
} // namespace onert
