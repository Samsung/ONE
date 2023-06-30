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
      throw std::runtime_error("train InferenceKernelGenerator : Not supported operation yet");
  }
}
} // namespace

InferenceKernelGenerator::InferenceKernelGenerator(
  const ir::Graph &graph, const std::shared_ptr<TensorBuilder> &tensor_builder,
  const std::shared_ptr<basic::TensorRegistry> &tensor_reg,
  const std::shared_ptr<backend::custom::IKernelBuilder> &kernel_builder,
  const std::shared_ptr<ExternalContext> &external_context)
  : basic::KernelGeneratorBase{graph},
    _ctx(graph.operands()), _operations_ctx{graph.operations()}, _current_layout{graph.layout()},
    _tensor_builder(tensor_builder), _tensor_reg{tensor_reg}, _kernel_builder(kernel_builder),
    _external_context(external_context)
{
  // DO NOTHING
}

std::unique_ptr<exec::FunctionSequence> InferenceKernelGenerator::generate(ir::OperationIndex ind)
{
  // TODO Generate FunctionSequence for backwarding as well
  auto ret = std::make_unique<exec::FunctionSequence>();
  ret->enableDynamicShapeInferer(false);

  const auto &op = _graph.operations().at(ind);
  op.accept(*this);
  ret->append(releaseFunction());

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

void InferenceKernelGenerator::visit(const ir::operation::Conv2D &node)
{
  using ir::operation::Conv2D;

  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(Conv2D::Input::INPUT)};
  const auto ker_index{node.getInputs().at(Conv2D::Input::KERNEL)};
  const auto bias_index{node.getInputs().at(Conv2D::Input::BIAS)};

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);
  auto ker_tensor = _tensor_reg->getPortableTensor(ker_index);
  auto bias_tensor = _tensor_reg->getPortableTensor(bias_index);

  const auto stride = node.param().stride;
  const auto activation = node.param().activation;
  const auto param_padding = node.param().padding;
  const auto dilation = node.param().dilation;
  auto fn = std::make_unique<ops::ConvolutionLayer>();

  if (_ctx.at(ifm_index).info().isDynamic() || _ctx.at(ker_index).info().isDynamic())
  {
    fn->configure(ifm_tensor, ker_tensor, bias_tensor, param_padding.type, param_padding.param.left,
                  param_padding.param.right, param_padding.param.top, param_padding.param.bottom,
                  stride.horizontal, stride.vertical, dilation.width_factor, dilation.height_factor,
                  activation, ofm_tensor);

    _return_fn = std::move(fn);
    return;
  }
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_layout);
  // Kernel format is [depth_out, kernel_height, kernel_width, depth_in].
  const auto &ker_shape = _ctx.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);

  const auto padding =
    ir::calculatePadding(param_padding, ifm_shape, ofm_shape, stride, ker_width, ker_height,
                         dilation.width_factor, dilation.height_factor);

  fn->configure(ifm_tensor, ker_tensor, bias_tensor, param_padding.type, padding.left,
                padding.right, padding.top, padding.bottom, stride.horizontal, stride.vertical,
                dilation.width_factor, dilation.height_factor, activation, ofm_tensor);

  _return_fn = std::move(fn);
}

void InferenceKernelGenerator::visit(const ir::operation::Pool2D &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Pool2D::Input::INPUT)};

  const auto kh = node.param().kh;
  const auto kw = node.param().kw;
  const auto stride = node.param().stride;
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_layout);
  const auto padding =
    ir::calculatePadding(node.param().padding, ifm_shape, ofm_shape, stride, kw, kh);
  const auto activation = node.param().activation;

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);

  auto fn = std::make_unique<ops::PoolLayer>();

  fn->configure(ifm_tensor, padding.left, padding.right, padding.top, padding.bottom,
                stride.horizontal, stride.vertical, kw, kh, activation, ofm_tensor,
                convertPoolType(node.param().op_type));

  _return_fn = std::move(fn);
}

KernelGenerator::KernelGenerator(const ir::train::TrainableGraph &tgraph,
                                 const std::shared_ptr<basic::TensorRegistry> &tensor_reg,
                                 const std::shared_ptr<basic::TensorRegistry> &grad_tensor_reg,
                                 const std::shared_ptr<ExternalContext> &external_context)
  : backend::train::KernelGeneratorBase{tgraph}, _current_layout{tgraph.layout()},
    _tensor_reg{tensor_reg}, _grad_tensor_reg{grad_tensor_reg}, _external_context(external_context)
{
  // DO NOTHING
}

std::unique_ptr<exec::train::TrainableFnSequence> KernelGenerator::generate(ir::OperationIndex ind)
{
  // TODO Generate TrainableFnSequence that can go backward as well
  auto ret = std::make_unique<exec::train::TrainableFnSequence>();

  const auto &op = _tgraph.operations().at(ind);
  // op.accept(*this);
  // ret->append(releaseFunction());

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

void KernelGenerator::visit(const ir::train::operation::Loss &)
{
  // TODO Generate kernel
}

} // namespace train
} // namespace backend
} // namespace onert
