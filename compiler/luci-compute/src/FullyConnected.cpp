/* Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci_compute/Types.h"
#include "luci_compute/FullyConnected.h"

#include "ConvertTypes.h"

#include <tensorflow/lite/kernels/internal/reference/fully_connected.h>

#include <cassert>
#include <cstdint>

namespace luci
{
namespace compute
{

namespace
{

} // namespace

bool FullyConnected::prepare(void)
{
  if (_input_shape.rank() < 1 || _weights_shape.rank() != 2)
    return false;

  auto const input_elems = element_count(&_input_shape);
  auto const weights_height = _weights_shape.dim(0).value();
  auto const weights_width = _weights_shape.dim(1).value();
  if (weights_height == 0 || weights_width == 0)
    return false;
  if (input_elems % weights_width != 0)
    return false;
  auto const batch_size = input_elems / weights_width;
  auto const num_units = weights_height;
  if (_bias_data)
  {
    if (element_count(&_bias_shape) != num_units)
      return false;
  }

  if (_keep_num_dims)
  {
    _output_shape.rank(_input_shape.rank());
    for (uint32_t i = 0; i < _input_shape.rank(); i++)
      _output_shape.dim(i) = _input_shape.dim(i);
    _output_shape.dim(_input_shape.rank() - 1) = num_units;
  }
  else
  {
    _output_shape.rank(2);
    _output_shape.dim(0) = batch_size;
    _output_shape.dim(1) = num_units;
  }

  return true;
}

void FullyConnected::compute(void)
{
  assert(_input_data != nullptr);
  assert(_weights_data != nullptr);
  // NOTE _bias_shape can be nullptr
  assert(_output_data != nullptr);

  // NOTE if this fails, structure may have changed
  static_assert(sizeof(compute::FullyConnectedParams) == sizeof(tflite::FullyConnectedParams));

  tflite::FullyConnectedParams params;

  // clang-format off
  params.input_offset             = _params.input_offset;
  params.weights_offset           = _params.weights_offset;
  params.output_offset            = _params.output_offset;
  params.output_multiplier        = _params.output_multiplier;
  params.output_shift             = _params.output_shift;
  params.quantized_activation_min = _params.quantized_activation_min;
  params.quantized_activation_max = _params.quantized_activation_max;
  params.float_activation_min     = _params.float_activation_min;
  params.float_activation_max     = _params.float_activation_max;
  params.lhs_cacheable            = _params.lhs_cacheable;
  params.rhs_cacheable            = _params.rhs_cacheable;
  params.weights_format           = tflite_weights_format(_params.weights_format);
  // clang-format on

  tflite::reference_ops::FullyConnected(
    params, tflite_shape(_input_shape), _input_data, tflite_shape(_weights_shape), _weights_data,
    tflite_shape(_bias_shape), _bias_data, tflite_shape(_output_shape), _output_data);
}

} // namespace compute
} // namespace luci
