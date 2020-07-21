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

#include "kernels/Conv2D.h"

#include "kernels/Utils.h"

#include <tensorflow/lite/kernels/internal/optimized/legacy_optimized_ops.h>

#include <stdexcept>
#include <thread>

namespace luci_interpreter
{
namespace kernels
{

Conv2D::Conv2D(const Tensor *input, const Tensor *filter, const Tensor *bias, Tensor *output,
               const Conv2DParams &params)
    : KernelWithParams<Conv2DParams>(params), _input(input), _filter(filter), _bias(bias),
      _output(output)
{
}

void Conv2D::configure()
{
  // TensorFlow Lite (as of v2.2.0) supports the following combinations of types:
  //     | input filter bias  output |
  // ----+---------------------------+
  // (1) | float float  float float  |
  // (2) | float int8   float float  | hybrid
  // (3) | uint8 uint8  int32 uint8  | quantized
  // (4) | int8  int8   int32 int8   | quantized per channel
  //
  // We only support (1) and (3) for now.
  if (_input->element_type() == DataType::FLOAT32 && _filter->element_type() == DataType::FLOAT32)
  {
    assert(_bias == nullptr || _bias->element_type() == DataType::FLOAT32);
  }
  else if (_input->element_type() == DataType::U8 && _filter->element_type() == DataType::U8)
  {
    assert(_bias == nullptr || _bias->element_type() == DataType::S32);
  }
  else
  {
    throw std::runtime_error("Unsupported type.");
  }
  assert(_output->element_type() == _input->element_type());

  const Shape &input_shape = _input->shape();
  const Shape &filter_shape = _filter->shape();
  assert(input_shape.num_dims() == 4 && filter_shape.num_dims() == 4);

  const int32_t batches = input_shape.dim(0);
  const int32_t input_height = input_shape.dim(1);
  const int32_t input_width = input_shape.dim(2);
  const int32_t output_depth = filter_shape.dim(0);
  const int32_t filter_height = filter_shape.dim(1);
  const int32_t filter_width = filter_shape.dim(2);
  assert(filter_shape.dim(3) == input_shape.dim(3));

  assert(_bias == nullptr ||
         (_bias->shape().num_dims() == 1 && _bias->shape().dim(0) == output_depth));

  const int32_t output_height =
      computeOutputSize(_params.padding, input_height, filter_height, _params.stride_height,
                        _params.dilation_height_factor);
  const int32_t output_width =
      computeOutputSize(_params.padding, input_width, filter_width, _params.stride_width,
                        _params.dilation_width_factor);

  _padding_height = computePadding(_params.stride_height, _params.dilation_height_factor,
                                   input_height, filter_height, output_height);
  _padding_width = computePadding(_params.stride_width, _params.dilation_width_factor, input_width,
                                  filter_width, output_width);

  _output->resize({batches, output_height, output_width, output_depth});

  // Allocate tensor for Im2Col, if needed.
  // The checks here should be aligned with the actual implementation.
  const bool need_dilated_im2col =
      _params.dilation_height_factor != 1 || _params.dilation_width_factor != 1;
  const bool need_non_dilated_im2col = _params.stride_height != 1 || _params.stride_width != 1 ||
                                       filter_height != 1 || filter_width != 1;
  const bool need_im2col = need_dilated_im2col || need_non_dilated_im2col;
  if (need_im2col)
  {
    const int input_depth = input_shape.dim(3);
    Shape im2col_shape{batches, output_height, output_width,
                       input_depth * filter_height * filter_width};
    _im2col =
        std::make_unique<Tensor>(_input->element_type(), im2col_shape, AffineQuantization{}, "");
  }
}

void Conv2D::execute() const
{
  switch (_input->element_type())
  {
    case DataType::FLOAT32:
      if (_filter->element_type() == DataType::FLOAT32)
      {
        evalFloat();
        break;
      }
      throw std::runtime_error("Unsupported type.");
    case DataType::U8:
      evalQuantized();
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

void Conv2D::evalFloat() const
{
  float activation_min{};
  float activation_max{};
  calculateActivationRange(_params.activation, &activation_min, &activation_max);

  tflite::ConvParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.dilation_height_factor = _params.dilation_height_factor;
  params.dilation_width_factor = _params.dilation_width_factor;
  params.float_activation_min = activation_min;
  params.float_activation_max = activation_max;

  tflite::optimized_ops::Conv(params, getTensorShape(_input), getTensorData<float>(_input),
                              getTensorShape(_filter), getTensorData<float>(_filter),
                              getTensorShape(_bias), getTensorData<float>(_bias),
                              getTensorShape(_output), getTensorData<float>(_output),
                              getTensorShape(_im2col.get()), getTensorData<float>(_im2col.get()));
}

void Conv2D::evalQuantized() const
{
  const auto input_scale = static_cast<double>(_input->scale());
  const auto filter_scale = static_cast<double>(_filter->scale());
  const auto output_scale = static_cast<double>(_output->scale());

  const double real_multiplier = input_scale * filter_scale / output_scale;
  int32_t output_multiplier{};
  int output_shift{};
  quantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);

  int32_t activation_min{};
  int32_t activation_max{};
  calculateActivationRangeQuantized(_params.activation, _output, &activation_min, &activation_max);

  tflite::ConvParams params{};
  params.padding_values.height = _padding_height;
  params.padding_values.width = _padding_width;
  params.stride_height = _params.stride_height;
  params.stride_width = _params.stride_width;
  params.dilation_height_factor = _params.dilation_height_factor;
  params.dilation_width_factor = _params.dilation_width_factor;
  // The kernel expects input and filter zero points to be negated.
  params.input_offset = -_input->zero_point();    // Note the '-'.
  params.weights_offset = -_filter->zero_point(); // Note the '-'.
  params.output_offset = _output->zero_point();
  params.output_multiplier = output_multiplier;
  params.output_shift = output_shift;
  params.quantized_activation_min = activation_min;
  params.quantized_activation_max = activation_max;

  // TODO This should only be done once (although it takes only a few microseconds).
  //  Also, the user should be able to adjust the number of threads.
  auto gemmlowp_context = std::make_unique<gemmlowp::GemmContext>();
  gemmlowp_context->set_max_num_threads(static_cast<int>(std::thread::hardware_concurrency()));

  tflite::optimized_ops::Conv(
      params, getTensorShape(_input), getTensorData<uint8_t>(_input), getTensorShape(_filter),
      getTensorData<uint8_t>(_filter), getTensorShape(_bias), getTensorData<int32_t>(_bias),
      getTensorShape(_output), getTensorData<uint8_t>(_output), getTensorShape(_im2col.get()),
      getTensorData<uint8_t>(_im2col.get()), gemmlowp_context.get());
}

} // namespace kernels
} // namespace luci_interpreter
