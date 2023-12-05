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

#include "KernelGenerator.h"

#include "ops/ConvolutionLayer.h"
#include "ops/FullyConnectedLayer.h"

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
namespace ruy
{

std::unique_ptr<exec::FunctionSequence> KernelGenerator::generate(ir::OperationIndex ind)
{
  auto ret = std::make_unique<exec::FunctionSequence>();

  assert(_tensor_builder->dynamicTensorManager());
  assert(_tensor_reg);

  // Prepare to handle dynamic tensors later
  auto dyn_ctx = std::make_shared<exec::FunctionSequence::DynamicTensorCtx>();
  {
    dyn_ctx->op = &_operations_ctx.at(ind);
    dyn_ctx->dynamic_shape_inferer = std::make_shared<exec::DynamicShapeInferer>(_ctx, _tensor_reg);
  }
  ret->dynamic_tensor_ctx(dyn_ctx);

  auto &op = _graph.operations().at(ind);
  op.accept(*this);
  assert(_return_fn); // _return_fn must have been generated
  ret->append(std::move(_return_fn));

  for (auto &&ind : (op.getInputs() | ir::Remove::UNDEFINED) + op.getOutputs())
  {
    auto portable_tensor = _tensor_reg->getPortableTensor(ind);
    if (portable_tensor)
    {
      assert(portable_tensor->layout() == ir::Layout::NHWC ||
             portable_tensor->layout() == ir::Layout::UNKNOWN);
    }

    auto tensor = _tensor_reg->getNativeTensor(ind);
    if (tensor)
    {
      tensor->increase_ref();
    }
  }
  return ret;
}

KernelGenerator::KernelGenerator(
  const ir::Graph &graph, const std::shared_ptr<TensorBuilder> &tensor_builder,
  const std::shared_ptr<basic::TensorRegistry> &tensor_reg,
  const std::shared_ptr<backend::custom::IKernelBuilder> &kernel_builder,
  const std::shared_ptr<ExternalContext> &external_context)
  : basic::KernelGeneratorBase{graph}, _ctx(graph.operands()), _operations_ctx{graph.operations()},
    _tensor_builder(tensor_builder), _tensor_reg{tensor_reg}, _kernel_builder(kernel_builder),
    _external_context(external_context)
{
  // DO NOTHING
}

void KernelGenerator::visit(const ir::operation::Conv2D &node)
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
                  activation, ofm_tensor, _external_context);

    _return_fn = std::move(fn);
    return;
  }
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_ctx.at(ifm_index).info().layout());
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_ctx.at(ofm_index).info().layout());
  // Kernel format is [depth_out, kernel_height, kernel_width, depth_in].
  const auto &ker_shape = _ctx.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);

  const auto padding =
    ir::calculatePadding(param_padding, ifm_shape, ofm_shape, stride, ker_width, ker_height,
                         dilation.width_factor, dilation.height_factor);

  fn->configure(ifm_tensor, ker_tensor, bias_tensor, param_padding.type, padding.left,
                padding.right, padding.top, padding.bottom, stride.horizontal, stride.vertical,
                dilation.width_factor, dilation.height_factor, activation, ofm_tensor,
                _external_context);

  _return_fn = std::move(fn);
}

void KernelGenerator::visit(const ir::operation::FullyConnected &node)
{
  using ir::operation::FullyConnected;

  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(FullyConnected::Input::INPUT)};
  const auto weight_index{node.getInputs().at(FullyConnected::Input::WEIGHT)};
  const auto bias_index{node.getInputs().at(FullyConnected::Input::BIAS)};
  const auto activation = node.param().activation;
  const auto weights_format = node.param().weights_format;

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto weight_tensor = _tensor_reg->getPortableTensor(weight_index);
  auto bias_tensor = bias_index.undefined() ? nullptr : _tensor_reg->getPortableTensor(bias_index);

  auto fn = std::make_unique<ops::FullyConnectedLayer>();

  fn->configure(input_tensor, weight_tensor, bias_tensor, activation, weights_format, output_tensor,
                _external_context);

  _return_fn = std::move(fn);
}

} // namespace ruy
} // namespace backend
} // namespace onert
