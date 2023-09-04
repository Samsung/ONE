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

#include "DepthwiseConvolutionLayer.h"

#include "cker/PortableTensorUtils.h"
#include <cker/operation/DepthwiseConv.h>

namespace onert
{
namespace backend
{
namespace cpu
{
namespace ops
{

void DepthwiseConvolutionLayer::convFloat32()
{
  float output_activation_min = 0, output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);

  nnfw::cker::DepthwiseConvParams op_params;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = _dilationWidth;
  op_params.dilation_height_factor = _dilationHeight;
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.depth_multiplier = _multiplier;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  nnfw::cker::DepthwiseConv<float, float>(
    op_params, getShape(_input), getBuffer<float>(_input), getShape(_kernel),
    getBuffer<float>(_kernel), getShape(_bias), getBuffer<float>(_bias), getShape(_output),
    getBuffer<float>(_output), _external_context->ruy_context());
}

void DepthwiseConvolutionLayer::convQ8uPerTensor()
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

  nnfw::cker::DepthwiseConvParams op_params;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = _dilationWidth;
  op_params.dilation_height_factor = _dilationHeight;
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.depth_multiplier = _multiplier;
  op_params.input_offset = -_input->data_zero_point();
  op_params.weights_offset = -_kernel->data_zero_point();
  op_params.output_offset = _output->data_zero_point();
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  nnfw::cker::DepthwiseConv<uint8_t, int32_t>(
    op_params, getShape(_input), getBuffer<uint8_t>(_input), getShape(_kernel),
    getBuffer<uint8_t>(_kernel), getShape(_bias), getBuffer<int32_t>(_bias), getShape(_output),
    getBuffer<uint8_t>(_output), _external_context->ruy_context());
}

void DepthwiseConvolutionLayer::convQ8uPerChannel()
{
  nnfw::cker::DepthwiseConvParams op_params;
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = _dilationWidth;
  op_params.dilation_height_factor = _dilationHeight;
  op_params.depth_multiplier = _multiplier;
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

  nnfw::cker::reference_integer_ops::DepthwiseConvPerChannel(
    op_params, _per_channel_output_multiplier.data(), _per_channel_output_shift.data(),
    getShape(_input), getBuffer<uint8_t>(_input), getShape(_kernel), getBuffer<uint8_t>(_kernel),
    _kernel->data_zero_points().data(), getShape(_bias), getBuffer<int32_t>(_bias),
    getShape(_output), getBuffer<uint8_t>(_output));
}

void DepthwiseConvolutionLayer::convQ8i()
{
  if (!_prepared)
  {
    prepareQ8i();
    _prepared = true;
  }

  int32_t output_activation_min = 0;
  int32_t output_activation_max = 0;
  CalculateActivationRangeQuantized(_activation, _output, &output_activation_min,
                                    &output_activation_max);

  nnfw::cker::DepthwiseConvParams op_params;
  op_params.padding_type = nnfw::cker::PaddingType::kSame;
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.depth_multiplier = _multiplier;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = _dilationWidth;
  op_params.dilation_height_factor = _dilationHeight;
  op_params.input_offset = -_input->data_zero_point();
  op_params.weights_offset = 0;
  op_params.output_offset = _output->data_zero_point();
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;

  nnfw::cker::optimized_integer_ops::DepthwiseConvPerChannel(
    op_params, _per_channel_output_multiplier.data(), _per_channel_output_shift.data(),
    getShape(_input), getBuffer<int8_t>(_input), getShape(_kernel), getBuffer<int8_t>(_kernel),
    getShape(_bias), getBuffer<int32_t>(_bias), getShape(_output), getBuffer<int8_t>(_output),
    _external_context->ruy_context());
}

void DepthwiseConvolutionLayer::convQ8iHybridPerChannel()
{
  if (!_prepared)
  {
    prepareQ8iHybridPerChannel();
    _prepared = true;
  }

  float output_activation_min = 0, output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);

  auto input_shape = getShape(_input);
  const int batch_size = input_shape.Dims(0);
  const int input_size = input_shape.FlatSize() / batch_size;

  auto scaling_factors_ptr = _input_scaling_factors.data();
  auto input_offsets_ptr = _input_offsets.data();

  for (int b = 0; b < batch_size; ++b)
  {
    const int offset = b * input_size;
    nnfw::cker::PortableAsymmetricQuantizeFloats(getBuffer<float>(_input) + offset, input_size,
                                                 _input_quantized.data() + offset,
                                                 &scaling_factors_ptr[b], &input_offsets_ptr[b]);
  }

  nnfw::cker::DepthwiseConvParams op_params;
  op_params.padding_values.width = _paddingLeft;
  op_params.padding_values.height = _paddingTop;
  op_params.depth_multiplier = _multiplier;
  op_params.stride_width = _strideWidth;
  op_params.stride_height = _strideHeight;
  op_params.dilation_width_factor = _dilationWidth;
  op_params.dilation_height_factor = _dilationHeight;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  nnfw::cker::reference_integer_ops::DepthwiseConvHybridPerChannel(
    op_params, _input_scaling_factors.data(), getShape(_input), _input_quantized.data(),
    getShape(_kernel), getBuffer<int8_t>(_kernel), getShape(_bias), getBuffer<float>(_bias),
    getShape(_output), getBuffer<float>(_output), _kernel->data_scales().data(),
    _input_offsets.data());
}

void DepthwiseConvolutionLayer::prepareQ8i()
{
  GetQuantizedConvolutionMultipliersAndShifts(
    _input->data_scale(), _output->data_scale(), _kernel->data_scales().data(),
    _kernel->data_scales().size(), getShape(_kernel).Dims(3), _per_channel_output_multiplier,
    _per_channel_output_shift);
}

void DepthwiseConvolutionLayer::prepareQ8uPerChannel()
{
  GetQuantizedConvolutionMultipliersAndShifts(
    _input->data_scale(), _output->data_scale(), _kernel->data_scales().data(),
    _kernel->data_scales().size(), getShape(_kernel).Dims(3), _per_channel_output_multiplier,
    _per_channel_output_shift);
}

void DepthwiseConvolutionLayer::prepareQ8iHybridPerChannel()
{
  // allocate memory for activation quantization.
  // - quantized values (int8_t type and same shape of original input)
  // - quantization params (= scale/zeropoint for each input)
  auto input_shape = getShape(_input);
  const int batch_size = input_shape.Dims(0);
  const int input_size = input_shape.FlatSize() / batch_size;
  _input_quantized.resize(input_size);
  // TODO: Optimize the case of batch_size = 1
  _input_scaling_factors.resize(batch_size);
  _input_offsets.resize(batch_size);
}

void DepthwiseConvolutionLayer::ensureQ8iHybridPerChannel()
{
  // ensure weight is per-channel quantized.
  int32_t kernel_input_channel = getShape(_kernel).Dims(3);
  // zero_points comes from flatbuffer vector. Its size is within uint32_t range.
  size_t kernel_zerop_cnt = _kernel->data_scales().size();
  // promote to int64_t to compare int32_t and uint32_t
  if ((int64_t)kernel_input_channel != (int64_t)kernel_zerop_cnt)
    throw std::runtime_error{"DConv2D hybrid supports only per-channel quantized weight."};
}

void DepthwiseConvolutionLayer::configure(
  const IPortableTensor *input, const IPortableTensor *kernel, const IPortableTensor *bias,
  const uint32_t paddingLeft, const uint32_t paddingRight, const uint32_t paddingTop,
  const uint32_t paddingBottom, const uint32_t strideWidth, const uint32_t strideHeight,
  const uint32_t multiplier, const uint32_t dilationWidth, const uint32_t dilationHeight,
  const ir::Activation activation, IPortableTensor *output,
  const std::shared_ptr<ExternalContext> &external_context)
{
  _input = input;
  _kernel = kernel;
  _bias = bias;
  _paddingLeft = paddingLeft;
  _paddingRight = paddingRight;
  _paddingTop = paddingTop;
  _paddingBottom = paddingBottom;
  _strideWidth = strideWidth;
  _strideHeight = strideHeight;
  _multiplier = multiplier;
  _dilationWidth = dilationWidth;
  _dilationHeight = dilationHeight;
  _activation = activation;
  _output = output;
  _external_context = external_context;
  _is_hybrid = _input->data_type() == OperandType::FLOAT32 &&
               _kernel->data_type() == OperandType::QUANT_INT8_SYMM;

  if (_is_hybrid)
  {
    ensureQ8iHybridPerChannel();
    prepareQ8iHybridPerChannel();
    _prepared = true;
  }
  else if (_input->data_type() == OperandType::QUANT_INT8_ASYMM)
  {
    if (_kernel->is_constant() && !_input->is_dynamic() && !_output->is_dynamic())
    {
      prepareQ8i();
      _prepared = true;
    }
  }
  else if (_input->data_type() == OperandType::QUANT_UINT8_ASYMM && _kernel->is_constant() &&
           !_input->is_dynamic() && !_output->is_dynamic())
  {
    const bool per_channel_quantized = _kernel->data_scales().size() > 1;
    if (per_channel_quantized)
    {
      prepareQ8uPerChannel();
      _prepared = true;
    }
  }
}

void DepthwiseConvolutionLayer::run()
{
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
    throw std::runtime_error{"DepthwiseConv: unsupported data type"};
  }
}

} // namespace ops
} // namespace cpu
} // namespace backend
} // namespace onert
