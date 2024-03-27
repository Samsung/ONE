/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kernels/FullyConnected.h"

#include "kernels/Utils.h"

#include "PALFullyConnected.h"

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

FullyConnected::FullyConnected(const Tensor *input, const Tensor *weights, const Tensor *bias,
                               Tensor *output, const FullyConnectedParams &params)
  : KernelWithParams<FullyConnectedParams>({input, weights, bias}, {output}, params)
{
}

void FullyConnected::configure()
{
  if (weights()->element_type() == DataType::U8)
  {
    LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::U8);
    LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::U8);
    LUCI_INTERPRETER_CHECK(!bias() || bias()->element_type() == DataType::S32)
  }
  else if (weights()->element_type() == DataType::FLOAT32)
  {
    LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(!bias() || bias()->element_type() == DataType::FLOAT32)
  }
  else if (weights()->element_type() == DataType::S8)
  {
    LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::S8);
    LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::S8);
    LUCI_INTERPRETER_CHECK(!bias() || bias()->element_type() == DataType::S32)
  }
  else if (weights()->element_type() == DataType::S4)
  {
    // TODO support other combinations when needed
    LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(!bias() || bias()->element_type() == DataType::FLOAT32)
  }
  else if (weights()->element_type() == DataType::U4)
  {
    // TODO support other combinations when needed
    LUCI_INTERPRETER_CHECK(input()->element_type() == DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(output()->element_type() == DataType::FLOAT32);
    LUCI_INTERPRETER_CHECK(!bias() || bias()->element_type() == DataType::FLOAT32)
  }
  else
  {
    throw std::runtime_error("luci-intp FullyConnected(1) Unsupported type.");
  }

  const Shape &input_shape = input()->shape();
  const Shape &weights_shape = weights()->shape();

  LUCI_INTERPRETER_CHECK(weights_shape.num_dims() == 2);
  LUCI_INTERPRETER_CHECK(bias() == nullptr ||
                         bias()->shape().num_elements() == weights_shape.dim(0));

  LUCI_INTERPRETER_CHECK(input_shape.num_elements() % weights_shape.dim(1) == 0);
  const int32_t batch_size = input_shape.num_elements() / weights_shape.dim(1);
  const int32_t num_units = weights_shape.dim(0);

  if (bias())
    LUCI_INTERPRETER_CHECK(bias()->shape().num_elements() == weights()->shape().dim(0));

  if (params().keep_num_dims == false)
  {
    output()->resize({batch_size, num_units});
  }
  else
  {
    luci_interpreter::Shape output_shape(input_shape.num_dims());
    for (int i = 0; i < input_shape.num_dims(); ++i)
      output_shape.dim(i) = input_shape.dim(i);
    output_shape.dim(input_shape.num_dims() - 1) = num_units;
    output()->resize(output_shape);
  }
}

void FullyConnected::execute() const
{
  const bool is_hybrid =
    (input()->element_type() == DataType::FLOAT32 &&
     (weights()->element_type() == DataType::S4 || weights()->element_type() == DataType::U4) &&
     output()->element_type() == DataType::FLOAT32 &&
     (!bias() || bias()->element_type() == DataType::FLOAT32));
  if (is_hybrid)
  {
    switch (weights()->element_type())
    {
      case DataType::S4:
        evalHybridWI4AF32();
        break;
      case DataType::U4:
        evalHybridWU4AF32();
        break;
      default:
        throw std::runtime_error("luci-intp FullyConnected(3) Unsupported type.");
    }
  }
  else
  {
    switch (input()->element_type())
    {
      case DataType::U8:
        evalQuantized();
        break;
      case DataType::S8:
        evalQuantizedS8();
        break;
      case DataType::FLOAT32:
        evalFloat();
        break;
      default:
        throw std::runtime_error("luci-intp FullyConnected(2) Unsupported type.");
    }
  }
}

void FullyConnected::evalFloat() const
{
  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  tflite::FullyConnectedParams params{};
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;
  params.weights_format = tflite::FullyConnectedWeightsFormat::kDefault;

  tflite::reference_ops::FullyConnected(
    params, getTensorShape(input()), getTensorData<float>(input()), getTensorShape(weights()),
    getTensorData<float>(weights()), getTensorShape(bias()), getTensorData<float>(bias()),
    getTensorShape(output()), getTensorData<float>(output()));
}

void FullyConnected::evalQuantized() const
{
  double real_multiplier = 0.0;
  int output_shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  int32_t output_multiplier;
  real_multiplier =
    getQuantizedConvolutionMultipler(input()->scale(), weights()->scale(), output()->scale());
  quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
  calculateActivationRangeQuantized(params().activation, output(), &output_activation_min,
                                    &output_activation_max);

  int32_t input_offset = -input()->zero_point();
  int32_t filter_offset = -weights()->zero_point();
  int32_t output_offset = output()->zero_point();

  tflite::FullyConnectedParams op_params{};
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.lhs_cacheable = false;
  op_params.rhs_cacheable = false;
  tflite::reference_ops::FullyConnected(
    op_params, getTensorShape(input()), getTensorData<uint8_t>(input()), getTensorShape(weights()),
    getTensorData<uint8_t>(weights()), getTensorShape(bias()), getTensorData<int32_t>(bias()),
    getTensorShape(output()), getTensorData<uint8_t>(output()));
}

void FullyConnected::evalQuantizedS8() const
{
  double real_multiplier = 0.0;
  int output_shift;
  int32_t output_activation_min;
  int32_t output_activation_max;
  int32_t output_multiplier;
  real_multiplier =
    getQuantizedConvolutionMultipler(input()->scale(), weights()->scale(), output()->scale());
  quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
  calculateActivationRangeQuantized(params().activation, output(), &output_activation_min,
                                    &output_activation_max);

  int32_t input_offset = -input()->zero_point();
  int32_t filter_offset = -weights()->zero_point();
  int32_t output_offset = output()->zero_point();

  tflite::FullyConnectedParams op_params{};
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = output_multiplier;
  op_params.output_shift = output_shift;
  op_params.quantized_activation_min = output_activation_min;
  op_params.quantized_activation_max = output_activation_max;
  op_params.lhs_cacheable = false;
  op_params.rhs_cacheable = false;
  luci_interpreter_pal::FullyConnected<int8_t>(
    op_params, getTensorShape(input()), getTensorData<int8_t>(input()), getTensorShape(weights()),
    getTensorData<int8_t>(weights()), getTensorShape(bias()), getTensorData<int32_t>(bias()),
    getTensorShape(output()), getTensorData<int8_t>(output()));
}

void FullyConnected::evalHybridWI4AF32() const
{
  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  tflite::FullyConnectedParams params{};
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;
  params.weights_format = tflite::FullyConnectedWeightsFormat::kDefault;

  const int8_t *weights_int4 = getTensorData<int8_t>(weights());
  float *weights_float = getTensorData<float>(scratch());
  const Shape &weights_shape = weights()->shape();
  for (int32_t i = 0; i < weights_shape.num_elements(); ++i)
  {
    // 1bit for sign, 3bit for value
    weights_float[i] = weights()->scale() * weights_int4[i];
  }
  tflite::reference_ops::FullyConnected(
    params, getTensorShape(input()), getTensorData<float>(input()), getTensorShape(scratch()),
    getTensorData<float>(scratch()), getTensorShape(bias()), getTensorData<float>(bias()),
    getTensorShape(output()), getTensorData<float>(output()));
}

void FullyConnected::evalHybridWU4AF32() const
{
  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  tflite::FullyConnectedParams params{};
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;
  params.weights_format = tflite::FullyConnectedWeightsFormat::kDefault;

  const auto *weights_uint4 = getTensorData<uint8_t>(weights());
  auto *weights_float = getTensorData<float>(scratch());
  const Shape &weights_shape = weights()->shape();
  const auto weights_scales = weights()->scales();
  const auto weights_zero_points = weights()->zero_points();
  const auto weights_quantized_dimension = weights()->quantized_dimension();
  LUCI_INTERPRETER_CHECK(weights_quantized_dimension == 0);
  if (weights_scales.size() == 1)
  {
    // Per tensor
    const auto scale = weights()->scale();
    const auto zero_point = weights()->zero_point();
    LUCI_INTERPRETER_CHECK(zero_point >= 0 and zero_point <= 15);
    for (int32_t i = 0; i < weights_shape.num_elements(); ++i)
    {
      weights_float[i] =
        scale * static_cast<float>(static_cast<int32_t>(weights_uint4[i]) - zero_point);
    }
  }
  else
  {
    // Per channel
    const int32_t quant_dim_size = weights_shape.dim(weights_quantized_dimension);

    size_t outer_dims_size = 1;
    size_t inner_dims_size = 1;
    for (int i = 0; i < weights_quantized_dimension; ++i)
      outer_dims_size *= weights_shape.dim(i);
    for (int i = weights_quantized_dimension + 1; i < weights_shape.num_dims(); ++i)
      inner_dims_size *= weights_shape.dim(i);

    for (size_t outer_it = 0; outer_it < outer_dims_size; ++outer_it)
      for (int32_t channel = 0; channel < quant_dim_size; ++channel)
      {
        int32_t zero_point = weights_zero_points[channel];
        LUCI_INTERPRETER_CHECK(zero_point >= 0 and zero_point <= 15);
        float scale = weights_scales[channel];
        size_t offset = inner_dims_size * (quant_dim_size * outer_it + channel);
        for (size_t inner_it = 0; inner_it < inner_dims_size; ++inner_it)
        {
          weights_float[offset + inner_it] =
            scale *
            static_cast<float>(static_cast<int32_t>(weights_uint4[offset + inner_it]) - zero_point);
        }
      }
  }

  tflite::reference_ops::FullyConnected(
    params, getTensorShape(input()), getTensorData<float>(input()), getTensorShape(scratch()),
    getTensorData<float>(scratch()), getTensorShape(bias()), getTensorData<float>(bias()),
    getTensorShape(output()), getTensorData<float>(output()));
}

} // namespace kernels
} // namespace luci_interpreter
