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

#include "ConvolutionLayer.h"
#include "OperationUtils.h"
#include "cker/PortableTensorUtils.h"

#include "../Tensor.h"
#include "ir/Padding.h"
#include <cker/operation/Conv.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{
ConvolutionLayer::ConvolutionLayer()
  : _input(nullptr), _kernel(nullptr), _bias(nullptr), _output(nullptr),
    _paddingType(ir::PaddingType::EXPLICIT), _paddingLeft(0), _paddingTop(0), _paddingRight(0),
    _paddingBottom(0), _strideWidth(0), _strideHeight(0), _dilationWidthFactor(1),
    _dilationHeightFactor(1), _activation(ir::Activation::NONE),
    _conv_kernel(new nnfw::cker::Conv()), _prepare(false)
{
  // DO NOTHING
}

ConvolutionLayer::~ConvolutionLayer() = default;

void ConvolutionLayer::convFloat32()
{
  float output_activation_min = 0, output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);

  nnfw::cker::ConvParams op_params;
  op_params.padding_type = getPaddingType(_paddingType);
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = _dilationWidthFactor;
  op_params.dilation_height_factor = _dilationHeightFactor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  nnfw::cker::Conv &kernel = *_conv_kernel;
  kernel(op_params, getShape(_input), getBuffer<float>(_input), getShape(_kernel),
         getBuffer<float>(_kernel), getShape(_bias), getBuffer<float>(_bias), getShape(_output),
         getBuffer<float>(_output));
}

void ConvolutionLayer::convQ8uPerTensor()
{
  int32_t output_activation_min = 0;
  int32_t output_activation_max = 0;
  CalculateActivationRangeQuantized(_activation, _output, &output_activation_min,
                                    &output_activation_max);

  double real_multiplier = 0.0;
  int32_t output_multiplier = 0;
  int32_t output_shift = 0;
  GetQuantizedConvolutionMultiplier(_input, _kernel, _bias, _output, &real_multiplier);
  QuantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);

  nnfw::cker::ConvParams op_params;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = _dilationWidthFactor;
  op_params.dilation_height_factor = _dilationHeightFactor;
  op_params.padding_type = getPaddingType(_paddingType);
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.input_offset = -_input->data_zero_point();
  op_params.weights_offset = -_kernel->data_zero_point();
  op_params.output_offset = _output->data_zero_point();
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.is_replaced_weights = true;

  nnfw::cker::Conv &kernel = *_conv_kernel;
  kernel(op_params, getShape(_input), getBuffer<uint8_t>(_input), getShape(_kernel),
         getBuffer<uint8_t>(_kernel), getShape(_bias), getBuffer<int32_t>(_bias), getShape(_output),
         getBuffer<uint8_t>(_output));
}

void ConvolutionLayer::convQ8uPerChannel()
{
  nnfw::cker::ConvParams op_params;
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = _dilationWidthFactor;
  op_params.dilation_height_factor = _dilationHeightFactor;
  op_params.input_offset = -_input->data_zero_point();
  op_params.output_offset = _output->data_zero_point();
  int32_t output_activation_min = 0;
  int32_t output_activation_max = 0;
  CalculateActivationRangeQuantized(_activation, _output, &output_activation_min,
                                    &output_activation_max);
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  // NOTE: The following fields of ConvParams are not used:
  // padding_type, weights_offset, output_{multiplier,shift}, float_activation_{min,max}

  nnfw::cker::Conv &kernel = *_conv_kernel;
  kernel(op_params, getShape(_input), getBuffer<uint8_t>(_input), getShape(_kernel),
         getBuffer<uint8_t>(_kernel), _kernel->data_zero_points().data(), getShape(_bias),
         getBuffer<int32_t>(_bias), getShape(_output), getBuffer<uint8_t>(_output));
}

void ConvolutionLayer::convQ8i()
{
  int32_t output_activation_min = 0;
  int32_t output_activation_max = 0;
  CalculateActivationRangeQuantized(_activation, _output, &output_activation_min,
                                    &output_activation_max);

  nnfw::cker::ConvParams op_params;
  op_params.input_offset = -_input->data_zero_point();
  op_params.output_offset = _output->data_zero_point();
  op_params.stride_height = _strideHeight;
  op_params.stride_width = _strideWidth;
  op_params.dilation_height_factor = _dilationHeightFactor;
  op_params.dilation_width_factor = _dilationWidthFactor;
  op_params.padding_values.height = _paddingTop;
  op_params.padding_values.width = _paddingLeft;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  nnfw::cker::Conv &kernel = *_conv_kernel;
  kernel(op_params, getShape(_input), reinterpret_cast<const int8_t *>(_input->buffer()),
         getShape(_kernel), reinterpret_cast<const int8_t *>(_kernel->buffer()), getShape(_bias),
         reinterpret_cast<const int32_t *>(_bias->buffer()), getShape(_output),
         reinterpret_cast<int8_t *>(_output->buffer()));
}

void ConvolutionLayer::convQ8iHybridPerChannel()
{
  float output_activation_min = 0;
  float output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);

  const int batch_size = getShape(_input).Dims(0);
  if (batch_size == 0)
    throw std::runtime_error{"Convolution input batch_size = 0"};
  auto input_shape = getShape(_input);
  const int input_size = input_shape.FlatSize() / batch_size;

  auto input_quantized_ptr = _hybrid_arena->input_quantized.data();
  auto input_scaling_factors_ptr = _hybrid_arena->input_scaling_factors.data();
  auto input_offsets_ptr = _hybrid_arena->input_offsets.data();
  for (int b = 0; b < batch_size; ++b)
  {
    const int offset = b * input_size;
    nnfw::cker::PortableAsymmetricQuantizeFloats(
      reinterpret_cast<const float *>(_input->buffer()) + offset, input_size,
      input_quantized_ptr + offset, &input_scaling_factors_ptr[b], &input_offsets_ptr[b]);
  }
  nnfw::cker::ConvParams op_params;
  op_params.padding_type = getPaddingType(_paddingType);
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = _dilationWidthFactor;
  op_params.dilation_height_factor = _dilationHeightFactor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  const auto *filter_per_channel_scales = _kernel->data_scales().data();
  nnfw::cker::reference::HybridConvPerChannel(
    op_params, input_scaling_factors_ptr, getShape(_input), input_quantized_ptr, getShape(_kernel),
    reinterpret_cast<const int8_t *>(_kernel->buffer()), getShape(_bias),
    reinterpret_cast<const float *>(_bias->buffer()), getShape(_output),
    reinterpret_cast<float *>(_output->buffer()), filter_per_channel_scales, input_offsets_ptr);
}

void ConvolutionLayer::configure(const IPortableTensor *input, const IPortableTensor *kernel,
                                 const IPortableTensor *bias, const ir::PaddingType paddingType,
                                 const uint32_t paddingLeft, const uint32_t paddingRight,
                                 const uint32_t paddingTop, const uint32_t paddingBottom,
                                 const uint32_t strideWidth, const uint32_t strideHeight,
                                 const uint32_t dilationWidthFactor,
                                 const uint32_t dilationHeightFactor,
                                 const ir::Activation activation, IPortableTensor *output)
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
  _is_hybrid = _input->data_type() == OperandType::FLOAT32 &&
               _kernel->data_type() == OperandType::QUANT_INT8_SYMM;
}

void ConvolutionLayer::run()
{
  prepare();
  if (_input->is_dynamic() || _kernel->is_dynamic())
  {
    const auto ifm_shape = _input->getShape().asFeature(_input->layout());
    const auto ofm_shape = _output->getShape().asFeature(_input->layout());
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
  if (_is_hybrid)
  {
    convQ8iHybridPerChannel();
  }
  else if (_input->data_type() == OperandType::FLOAT32)
  {
    convFloat32();
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM)
  {
    const bool per_channel_quantized = _kernel->data_scales().size() > 1;
    if (per_channel_quantized)
      convQ8uPerChannel();
    else
      convQ8uPerTensor();
  }
  else if (_input->data_type() == OperandType::QUANT_INT8_ASYMM)
  {
    convQ8i();
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

  if (_is_hybrid)
  {
    // ensure weight is per-channel quantized.
    int32_t kernel_input_channel = getShape(_kernel).Dims(3);
    // zero_points comes from flatbuffer vector. Its size is within uint32_t range.
    size_t kernel_zerop_cnt = _kernel->data_scales().size();
    // promote to int64_t to compare int32_t and uint32_t
    if ((int64_t)kernel_input_channel != (int64_t)kernel_zerop_cnt)
      throw std::runtime_error{"DConv2D hybrid supports only per-channel quantized weight."};

    // allocate memory for activation quantization.
    // - quantized values (int8_t type and same shape of original input)
    // - quantization params (= scale/zeropoint for each input)
    auto input_shape = getShape(_input);
    const int batch_size = input_shape.Dims(0);
    const int input_size = input_shape.FlatSize() / batch_size;
    _hybrid_arena = std::make_unique<nnfw::cker::ConvHybridTempArena>(batch_size, input_size);
    _prepare = true;
    return;
  }

  nnfw::cker::Conv &kernel = *_conv_kernel;
  if (_input->data_type() == OperandType::FLOAT32 && _kernel->is_constant())
  {
    bool is_transposed = false;
    kernel.prepareF32(getShape(_kernel), getBuffer<float>(_kernel), getPaddingType(_paddingType),
                      is_transposed, _dilationWidthFactor, _dilationHeightFactor);

    // Decrease reference of _kernel(weights) only when _kernel is constant
    if (is_transposed)
    {
      auto kernel_tensor = dynamic_cast<const Tensor *>(_kernel);
      if (kernel_tensor)
        // TODO Remove const_cast
        const_cast<Tensor *>(kernel_tensor)->decrease_ref();
    }
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM && _kernel->is_constant() &&
           !_input->is_dynamic() && !_output->is_dynamic())
  {
    const bool per_channel_quantized = _kernel->data_scales().size() > 1;
    if (per_channel_quantized)
    {
      GetQuantizedConvolutionMultipliersAndShifts(
        _input->data_scale(), _output->data_scale(), _kernel->data_scales().data(),
        _kernel->data_scales().size(), getShape(_kernel).Dims(0),
        kernel.per_channel_output_multiplier(), kernel.per_channel_output_shift());
    }
    else
    {
      kernel.prepareQ8uPerTensor(getShape(_input), getShape(_kernel), getShape(_output),
                                 _strideWidth, _strideHeight, _dilationWidthFactor,
                                 _dilationHeightFactor);
    }
  }
  else if (_input->data_type() == OperandType::QUANT_INT8_ASYMM)
  {
    if (_kernel->is_constant() && !_input->is_dynamic() && !_output->is_dynamic())
    {
      GetQuantizedConvolutionMultipliersAndShifts(
        _input->data_scale(), _output->data_scale(), _kernel->data_scales().data(),
        _kernel->data_scales().size(), getShape(_kernel).Dims(0),
        kernel.per_channel_output_multiplier(), kernel.per_channel_output_shift());
    }
    else
    {
      throw std::runtime_error{"Conv2D: Int8 dynamic weight is not supported"};
    }
  }
  _prepare = true;
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
