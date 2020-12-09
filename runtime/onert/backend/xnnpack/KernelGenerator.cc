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
#include "ops/DepthwiseConvolutionLayer.h"
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
namespace xnnpack
{

KernelGenerator::KernelGenerator(
  const ir::Operands &operands_ctx, const ir::Operations &operations_ctx,
  const std::shared_ptr<TensorBuilder> &tensor_builder,
  const std::shared_ptr<cpu_common::TensorRegistry> &tensor_reg,
  const std::shared_ptr<backend::custom::IKernelBuilder> &kernel_builder,
  const std::shared_ptr<ExternalContext> &external_context)
  : _ctx(operands_ctx), _operations_ctx{operations_ctx},
    _tensor_builder(tensor_builder), _tensor_reg{tensor_reg}, _kernel_builder(kernel_builder),
    _current_layout(ir::Layout::UNKNOWN), _external_context(external_context)
{
  // DO NOTHING
}

void KernelGenerator::visit(const ir::OpSequence &op_seq)
{
  assert(!_return_fn_seq);
  assert(_tensor_builder->dynamicTensorManager());
  assert(_tensor_reg);

  auto dyn_shape_inferer = std::make_shared<exec::DynamicShapeInferer>(_ctx, _tensor_reg);

  _return_fn_seq = std::make_unique<exec::FunctionSequence>();

  // Prepare to handle dynamic tensors later
  auto dyn_ctx = std::make_shared<exec::FunctionSequence::DynamicTensorCtx>();
  {
    dyn_ctx->op_seq = &op_seq;
    dyn_ctx->operations = &_operations_ctx;
    dyn_ctx->dynamic_shape_inferer = std::move(dyn_shape_inferer);
    dyn_ctx->dynamic_tensor_manager = _tensor_builder->dynamicTensorManager();

    _return_fn_seq->dynamic_tensor_ctx(dyn_ctx);
  }

  _current_layout = op_seq.getLayout();
  for (const auto &operation_idx : op_seq.operations())
  {
    const auto &node = _operations_ctx.at(operation_idx);
    node.accept(*this);
    _return_fn_seq->append(releaseFunction());

    for (const auto &ind : (node.getInputs() | ir::Remove::UNDEFINED) + node.getOutputs())
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
  }
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
  auto fn = std::make_unique<ops::ConvolutionLayer>(_external_context);

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

void KernelGenerator::visit(const ir::operation::DepthwiseConv2D &node)
{
  using ir::operation::DepthwiseConv2D;

  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(DepthwiseConv2D::Input::INPUT)};
  const auto ker_index{node.getInputs().at(DepthwiseConv2D::Input::KERNEL)};
  const auto bias_index{node.getInputs().at(DepthwiseConv2D::Input::BIAS)};

  const auto stride = node.param().stride;
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature(_current_layout);
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature(_current_layout);
  // Kernel format is [1, kernel_height, kernel_width, depth_out].
  const auto &ker_shape = _ctx.at(ker_index).shape();
  const auto ker_height = ker_shape.dim(1);
  const auto ker_width = ker_shape.dim(2);
  const auto dilation_width = node.param().dilation.width_factor;
  const auto dilation_height = node.param().dilation.height_factor;
  const auto param_padding = node.param().padding;
  const auto padding = ir::calculatePadding(param_padding, ifm_shape, ofm_shape, stride, ker_width,
                                            ker_height, dilation_width, dilation_height);
  const auto multiplier = node.param().multiplier;
  const auto activation = node.param().activation;

  auto ofm_tensor = _tensor_reg->getPortableTensor(ofm_index);
  auto ifm_tensor = _tensor_reg->getPortableTensor(ifm_index);
  auto ker_tensor = _tensor_reg->getPortableTensor(ker_index);
  auto bias_tensor = _tensor_reg->getPortableTensor(bias_index);

  auto fn = std::make_unique<ops::DepthwiseConvolutionLayer>(_external_context);

  fn->configure(ifm_tensor, ker_tensor, bias_tensor, param_padding.type, padding.left,
                padding.right, padding.top, padding.bottom, stride.horizontal, stride.vertical,
                multiplier, dilation_width, dilation_height, activation, ofm_tensor);

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

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto input_tensor = _tensor_reg->getPortableTensor(input_index);
  auto weight_tensor = _tensor_reg->getPortableTensor(weight_index);
  auto bias_tensor = bias_index.undefined() ? nullptr : _tensor_reg->getPortableTensor(bias_index);

  auto fn = std::make_unique<ops::FullyConnectedLayer>(_external_context);

  fn->configure(input_tensor, weight_tensor, bias_tensor, activation, output_tensor);

  _return_fn = std::move(fn);
}

} // namespace xnnpack
} // namespace backend
} // namespace onert
