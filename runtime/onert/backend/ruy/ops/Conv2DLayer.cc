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

#include "Conv2DLayer.h"

#include "../KernelGenerator.h"
#include "../Tensor.h"
#include "../Validator.h"

#include "ir/Padding.h"

namespace onert::backend::ruy
{

void Validator::visit(const ir::operation::Conv2D &) { _supported = true; }

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
  const auto &param_padding = node.param().padding;
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
  const auto ifm_shape = _ctx.at(ifm_index).shape().asFeature();
  const auto ofm_shape = _ctx.at(ofm_index).shape().asFeature();
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

} // namespace onert::backend::ruy

namespace onert::backend::ruy::ops
{
ConvolutionLayer::ConvolutionLayer()
  : _input(nullptr), _kernel(nullptr), _bias(nullptr), _output(nullptr),
    _paddingType(ir::PaddingType::EXPLICIT), _paddingLeft(0), _paddingTop(0), _paddingRight(0),
    _paddingBottom(0), _strideWidth(0), _strideHeight(0), _dilationWidthFactor(1),
    _dilationHeightFactor(1), _activation(ir::Activation::NONE),
    _conv_kernel(new nnfw::ruy::Conv()), _prepare(false)
{
  // DO NOTHING
}

ConvolutionLayer::~ConvolutionLayer() = default;

void ConvolutionLayer::convFloat32()
{
  float output_activation_min = 0, output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);

  nnfw::ruy::ConvParams op_params;
  op_params.padding_type = getPaddingType(_paddingType);
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = _dilationWidthFactor;
  op_params.dilation_height_factor = _dilationHeightFactor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  nnfw::ruy::Conv &kernel = *_conv_kernel;
  kernel(op_params, getTensorShape(_input), reinterpret_cast<const float *>(_input->buffer()),
         getTensorShape(_kernel), reinterpret_cast<const float *>(_kernel->buffer()),
         getTensorShape(_bias), reinterpret_cast<const float *>(_bias->buffer()),
         getTensorShape(_output), reinterpret_cast<float *>(_output->buffer()),
         _external_context->ruy_context());
}

void ConvolutionLayer::configure(const IPortableTensor *input, const IPortableTensor *kernel,
                                 const IPortableTensor *bias, const ir::PaddingType paddingType,
                                 const uint32_t paddingLeft, const uint32_t paddingRight,
                                 const uint32_t paddingTop, const uint32_t paddingBottom,
                                 const uint32_t strideWidth, const uint32_t strideHeight,
                                 const uint32_t dilationWidthFactor,
                                 const uint32_t dilationHeightFactor,
                                 const ir::Activation activation, IPortableTensor *output,
                                 const std::shared_ptr<ExternalContext> &external_context)
{
  _input = input;
  _kernel = kernel;
  _bias = bias;
  _paddingType = paddingType;
  _paddingLeft = paddingLeft;
  _paddingRight = paddingRight;
  _paddingTop = paddingTop;
  _paddingBottom = paddingBottom;
  _strideWidth = strideWidth;
  _strideHeight = strideHeight;
  _dilationWidthFactor = dilationWidthFactor;
  _dilationHeightFactor = dilationHeightFactor;
  _activation = activation;
  _output = output;
  _external_context = external_context;
}

void ConvolutionLayer::run()
{
  prepare();

  if (_input->is_dynamic() || _kernel->is_dynamic())
  {
    const auto ifm_shape = _input->getShape().asFeature();
    const auto ofm_shape = _output->getShape().asFeature();
    // Kernel format is [depth_out, kernel_height, kernel_width, depth_in].
    const auto ker_shape = _kernel->getShape();
    const auto ker_height = ker_shape.dim(1);
    const auto ker_width = ker_shape.dim(2);

    ir::Stride stride;
    stride.vertical = _strideWidth;
    stride.horizontal = _strideWidth;

    ir::Padding param_padding;
    param_padding.type = _paddingType;
    param_padding.param.left = _paddingLeft;
    param_padding.param.right = _paddingRight;
    param_padding.param.top = _paddingTop;
    param_padding.param.bottom = _paddingBottom;

    const auto padding =
      ir::calculatePadding(param_padding, ifm_shape, ofm_shape, stride, ker_width, ker_height,
                           _dilationWidthFactor, _dilationHeightFactor);

    _paddingLeft = padding.left;
    _paddingRight = padding.right;
    _paddingTop = padding.top;
    _paddingBottom = padding.bottom;
  }
  if (_input->data_type() == OperandType::FLOAT32)
  {
    convFloat32();
  }
  else
  {
    throw std::runtime_error{"Conv: unsupported data type"};
  }
}

void ConvolutionLayer::prepare()
{
  if (_prepare)
    return;

  nnfw::ruy::Conv &kernel = *_conv_kernel;
  if (_input->data_type() == OperandType::FLOAT32 && _kernel->is_constant())
  {
    kernel.prepare(getTensorShape(_input), getTensorShape(_kernel), getTensorShape(_output),
                   _strideWidth, _strideHeight, _dilationWidthFactor, _dilationHeightFactor);
  }
  _prepare = true;
}

} // namespace onert::backend::ruy::ops
