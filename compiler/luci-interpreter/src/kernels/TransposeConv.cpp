/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#include "kernels/TransposeConv.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/reference/transpose_conv.h>

#include <stdexcept>
#include <limits> // std::numeric_limits

namespace luci_interpreter
{

namespace kernels
{

TransposeConv::TransposeConv(const Tensor *output_shape, const Tensor *filter, const Tensor *input,
                             const Tensor *bias, Tensor *output, Tensor *scratch_tensor,
                             const TransposeConvParams &params)
  : KernelWithParams<TransposeConvParams>({output_shape, filter, input, bias},
                                          {output, scratch_tensor}, params)
{
}

TransposeConv::~TransposeConv()
{
  // Define destructor here, to delete vector of qunatized multipliers properly
}

void TransposeConv::configure()
{
  assert(output_shape()->shape().num_dims() == 1);
  assert(input()->shape().num_dims() == 4);
  assert(filter()->shape().num_dims() == 4);
  assert(input()->element_type() == DataType::FLOAT32 || input()->element_type() == DataType::U8 ||
         input()->element_type() == DataType::S16);
  assert(input()->element_type() == output()->element_type());
  assert(input()->shape().dim(3) == filter()->shape().dim(3));

  const int num_dims = output_shape()->shape().dim(0);
  Shape out_shape(num_dims);
  const auto *shape_data = getTensorData<int32_t>(output_shape());
  for (int i = 0; i < num_dims; i++)
    out_shape.dim(i) = shape_data[i];
  output()->resize(out_shape);

  const int32_t filter_height = filter()->shape().dim(1);
  const int32_t filter_width = filter()->shape().dim(2);
  const int32_t output_height = out_shape.dim(1);
  const int32_t output_width = out_shape.dim(2);

  const int32_t unused_output_height =
    computeOutputSize(params().padding, output_height, filter_height, params().stride_height, 1);
  const int32_t unused_output_width =
    computeOutputSize(params().padding, output_width, filter_width, params().stride_width, 1);

  _padding_height =
    computePadding(params().stride_height, 1, output_height, filter_height, unused_output_height);
  _padding_width =
    computePadding(params().stride_width, 1, output_width, filter_width, unused_output_width);

  if (input()->element_type() == DataType::U8 || input()->element_type() == DataType::S16)
  {
    auto scratch_tensor = getOutputTensors()[1];
    scratch_tensor->resize(output()->shape());
    const std::vector<double> real_multipliers =
      getQuantizedConvolutionMultiplers(input()->scale(), filter()->scales(), output()->scale());

    _quant_multipliers = quantizeMultipliers(real_multipliers);
  }
  else
  {
    auto scratch_tensor = getOutputTensors()[1];
    scratch_tensor->set_allocatable(false);
  }
}

void TransposeConv::execute() const
{
  switch (input()->element_type())
  {
    case DataType::FLOAT32:
      evalFloat();
      break;
    case DataType::U8:
      if (filter()->scales().size() == 1)
      {
        evalQuantized();
      }
      else if (filter()->scales().size() > 1)
      {
        LUCI_INTERPRETER_CHECK(filter()->shape().num_dims() == 4);
        LUCI_INTERPRETER_CHECK(filter()->scales().size() ==
                               static_cast<size_t>(filter()->shape().dim(0)));
        evalQuantizedPerChannel();
      }
      break;
    case DataType::S16:
      evalQuantizedS16();
      break;
    default:
      throw std::runtime_error("luci-intp TransposeConv Unsupported type.");
  }
}

void TransposeConv::evalFloat() const
{
  float activation_min{};
  float activation_max{};
  // TODO support activation
  assert(_params.activation == Activation::NONE);
  calculateActivationRange(Activation::NONE, &activation_min, &activation_max);

  tflite::ConvParams op_params{};
  op_params.padding_type = tflite::PaddingType::kSame;
  op_params.padding_values.height = _padding_height;
  op_params.padding_values.width = _padding_width;
  op_params.stride_height = params().stride_height;
  op_params.stride_width = params().stride_width;
  op_params.float_activation_min = activation_min;
  op_params.float_activation_max = activation_max;
  tflite::reference_ops::TransposeConv(op_params,                                                //
                                       getTensorShape(input()), getTensorData<float>(input()),   //
                                       getTensorShape(filter()), getTensorData<float>(filter()), //
                                       getTensorShape(bias()), getTensorData<float>(bias()),     //
                                       getTensorShape(output()), getTensorData<float>(output()), //
                                       tflite::RuntimeShape(), nullptr);
}

void TransposeConv::evalQuantized() const
{
  tflite::ConvParams op_params{};
  op_params.padding_type = tflite::PaddingType::kSame;
  op_params.padding_values.height = _padding_height;
  op_params.padding_values.width = _padding_width;
  op_params.stride_height = params().stride_height;
  op_params.stride_width = params().stride_width;
  // The kernel expects input and filter zero points to be negated.
  op_params.input_offset = -input()->zero_point();    // Note the '-'.
  op_params.weights_offset = -filter()->zero_point(); // Note the '-'.
  op_params.output_offset = output()->zero_point();
  op_params.output_multiplier = _quant_multipliers[0].multiplier;
  op_params.output_shift = _quant_multipliers[0].shift;
  op_params.quantized_activation_min = std::numeric_limits<uint8_t>::min();
  op_params.quantized_activation_max = std::numeric_limits<uint8_t>::max();

  auto scratch_tensor = getOutputTensors()[1];

  tflite::reference_ops::TransposeConv(op_params,                                                //
                                       getTensorShape(input()), getTensorData<uint8>(input()),   //
                                       getTensorShape(filter()), getTensorData<uint8>(filter()), //
                                       getTensorShape(bias()), getTensorData<int32_t>(bias()),   //
                                       getTensorShape(output()), getTensorData<uint8>(output()), //
                                       tflite::RuntimeShape(), nullptr,                          //
                                       getTensorData<int32_t>(scratch_tensor));
}

void TransposeConv::evalQuantizedPerChannel() const
{
  const auto *input_data = getTensorData<uint8_t>(input());
  const auto *filter_data = getTensorData<uint8_t>(filter());
  const auto *bias_data = getTensorData<int32_t>(bias());
  auto *output_data = getTensorData<uint8_t>(output());

  auto scratch_tensor = getOutputTensors()[1];
  auto *scratch_data = getTensorData<int32_t>(scratch_tensor);

  const Shape &input_shape = input()->shape();
  const Shape &filter_shape = filter()->shape();
  const Shape &output_shape = output()->shape();

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t input_depth = input_shape.dim(3);
  const int32_t output_depth = filter_shape.dim(0);
  const int32_t filter_height = filter_shape.dim(1);
  const int32_t filter_width = filter_shape.dim(2);
  const int32_t output_height = output_shape.dim(1);
  const int32_t output_width = output_shape.dim(2);

  const int32_t stride_height = _params.stride_height;
  const int32_t stride_width = _params.stride_width;

  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(Activation::NONE, output(), &activation_min, &activation_max);

  std::memset(scratch_data, 0, scratch_tensor->shape().num_elements() * sizeof(int32_t));

  BroadcastableWrapper<ChannelQuantMultipliers> output_multipliers(_quant_multipliers);
  for (int32_t batch = 0; batch < batches; ++batch)
  {
    for (int32_t in_y = 0; in_y < input_height; ++in_y)
    {
      for (int32_t in_x = 0; in_x < input_width; ++in_x)
      {
        for (int32_t in_c = 0; in_c < input_depth; ++in_c)
        {
          const int32_t out_y_origin = in_y * stride_height - _padding_height;
          const int32_t out_x_origin = in_x * stride_width - _padding_width;
          for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            for (int32_t filter_x = 0; filter_x < filter_width; ++filter_x)
            {
              const int32_t out_x = out_x_origin + filter_x;
              const int32_t out_y = out_y_origin + filter_y;
              if ((out_y >= 0 && out_y < output_height) && (out_x >= 0 && out_x < output_width))
              {
                for (int32_t out_c = 0; out_c < output_depth; ++out_c)
                {
                  const uint8_t input_val =
                    input_data[calcOffset(input_shape, batch, in_y, in_x, in_c)];
                  const uint8_t filter_val =
                    filter_data[calcOffset(filter_shape, out_c, filter_y, filter_x, in_c)];
                  scratch_data[calcOffset(output_shape, batch, out_y, out_x, out_c)] +=
                    static_cast<int32_t>(input_val - input()->zero_point()) *
                    static_cast<int32_t>(filter_val - filter()->zero_points()[out_c]);
                }
              }
            }
          }
        }
      }
    }
    for (int32_t out_y = 0; out_y < output_height; ++out_y)
    {
      for (int32_t out_x = 0; out_x < output_width; ++out_x)
      {
        for (int32_t out_c = 0; out_c < output_depth; ++out_c)
        {
          int32_t acc = scratch_data[calcOffset(output_shape, batch, out_y, out_x, out_c)];
          if (bias_data)
          {
            acc += bias_data[out_c];
          }

          int32_t scaled_acc = tflite::MultiplyByQuantizedMultiplier(
            acc, output_multipliers[out_c].multiplier, output_multipliers[out_c].shift);

          scaled_acc += output()->zero_point();
          scaled_acc = std::max(scaled_acc, activation_min);
          scaled_acc = std::min(scaled_acc, activation_max);

          output_data[calcOffset(output_shape, batch, out_y, out_x, out_c)] = scaled_acc;
        }
      }
    }
  }
}

void TransposeConv::evalQuantizedS16() const
{
  const auto *input_data = getTensorData<int16_t>(input());
  const auto *filter_data = getTensorData<int16_t>(filter());
  const auto *bias_data = getTensorData<int64_t>(bias());
  auto *output_data = getTensorData<int16_t>(output());

  auto scratch_tensor = getOutputTensors()[1];
  auto *scratch_data = getTensorData<int64_t>(scratch_tensor);

  const Shape &input_shape = input()->shape();
  const Shape &filter_shape = filter()->shape();
  const Shape &output_shape = output()->shape();

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t input_depth = input_shape.dim(3);
  const int32_t output_depth = filter_shape.dim(0);
  const int32_t filter_height = filter_shape.dim(1);
  const int32_t filter_width = filter_shape.dim(2);
  const int32_t output_height = output_shape.dim(1);
  const int32_t output_width = output_shape.dim(2);

  const int32_t stride_height = _params.stride_height;
  const int32_t stride_width = _params.stride_width;

  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(Activation::NONE, output(), &activation_min, &activation_max);

  std::memset(scratch_data, 0, scratch_tensor->shape().num_elements() * sizeof(int64_t));

  BroadcastableWrapper<ChannelQuantMultipliers> output_multipliers(_quant_multipliers);
  for (int32_t batch = 0; batch < batches; ++batch)
  {
    for (int32_t in_y = 0; in_y < input_height; ++in_y)
    {
      for (int32_t in_x = 0; in_x < input_width; ++in_x)
      {
        for (int32_t in_c = 0; in_c < input_depth; ++in_c)
        {
          const int32_t out_y_origin = in_y * stride_height - _padding_height;
          const int32_t out_x_origin = in_x * stride_width - _padding_width;
          for (int32_t filter_y = 0; filter_y < filter_height; ++filter_y)
          {
            for (int32_t filter_x = 0; filter_x < filter_width; ++filter_x)
            {
              const int32_t out_x = out_x_origin + filter_x;
              const int32_t out_y = out_y_origin + filter_y;
              if ((out_y >= 0 && out_y < output_height) && (out_x >= 0 && out_x < output_width))
              {
                for (int32_t out_c = 0; out_c < output_depth; ++out_c)
                {
                  const int16_t input_val =
                    input_data[calcOffset(input_shape, batch, in_y, in_x, in_c)];
                  const int16_t filter_val =
                    filter_data[calcOffset(filter_shape, out_c, filter_y, filter_x, in_c)];
                  scratch_data[calcOffset(output_shape, batch, out_y, out_x, out_c)] +=
                    static_cast<int64_t>(input_val) * static_cast<int64_t>(filter_val);
                }
              }
            }
          }
        }
      }
    }
    for (int32_t out_y = 0; out_y < output_height; ++out_y)
    {
      for (int32_t out_x = 0; out_x < output_width; ++out_x)
      {
        for (int32_t out_c = 0; out_c < output_depth; ++out_c)
        {
          int64_t acc = scratch_data[calcOffset(output_shape, batch, out_y, out_x, out_c)];
          if (bias_data)
          {
            acc += bias_data[out_c];
          }
          int32_t scaled_acc = tflite::MultiplyByQuantizedMultiplier(
            acc, output_multipliers[out_c].multiplier, output_multipliers[out_c].shift);

          scaled_acc = std::max(scaled_acc, activation_min);
          scaled_acc = std::min(scaled_acc, activation_max);

          output_data[calcOffset(output_shape, batch, out_y, out_x, out_c)] = scaled_acc;
        }
      }
    }
  }
}

} // namespace kernels
} // namespace luci_interpreter
