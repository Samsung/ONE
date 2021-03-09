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

/*
 * Copyright (c) 2017-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "arm_compute/runtime/NEON/functions/NEFullyConnectedHybridLayer.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include <algorithm>
#include <cmath>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

namespace
{
Status validate_mm(const ITensorInfo &input, const ITensorInfo &weights, const ITensorInfo &output)
{
  ARM_COMPUTE_RETURN_ON_ERROR(
    NEGEMMLowpMatrixMultiplyCore::validate(&input, &weights, nullptr, &output));

  return Status{};
}
} // namespace

void NEFullyConnectedHybridLayerReshapeWeights::configure(const ITensor *input, ITensor *output)
{
  auto k = std::make_unique<NETransposeKernel>();
  k->configure(input, output);
  _kernel = std::move(k);
}

Status NEFullyConnectedHybridLayerReshapeWeights::validate(const ITensorInfo *input,
                                                           const ITensorInfo *output)
{
  return NETransposeKernel::validate(input, output);
}

NEFullyConnectedHybridLayer::NEFullyConnectedHybridLayer(
  std::shared_ptr<IMemoryManager> memory_manager)
  : _memory_group(std::move(memory_manager)), _reshape_weights_function(), _quant_input_kernel(),
    _mm_gemmlowp(), /*_accumulate_biases_kernel(),*/ _reshape_weights_output(), _quantized_input(),
    _scale_factor(), _original_weights(nullptr), _are_weights_reshaped(false),
    _accumulate_biases(false), _is_prepared(false)
{
}

void NEFullyConnectedHybridLayer::configure_mm(const ITensor *input, const ITensor *weights,
                                               ITensor *output)
{
  ARM_COMPUTE_ERROR_ON(input->info()->dimension(0) != weights->info()->dimension(1));

  // Configure gemmlowp function
  _mm_gemmlowp.configure(input, weights, nullptr, output);
}

void NEFullyConnectedHybridLayer::configure(const ITensor *input, const ITensor *weights,
                                            const ITensor *biases, ITensor *output,
                                            FullyConnectedLayerInfo fc_info)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

  // Perform validate step
  ARM_COMPUTE_ERROR_THROW_ON(NEFullyConnectedHybridLayer::validate(
    input->info(), weights->info(), biases != nullptr ? biases->info() : nullptr, output->info(),
    fc_info));

  _are_weights_reshaped = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;
  _accumulate_biases = false;
  _original_weights = weights;

  // Configure accumulate biases kernel for non quantized asymmetric types
  if (biases != nullptr)
  {
    _accumulate_biases = true;

    // Configure accumulate biases kernel
    // _accumulate_biases_kernel.configure(output, biases);
  }

  // With the Fully Connected layer we can have 4 different cases:
  //  1) Convolution layer -> Fully Connected layer without batches
  //  2) Fully Connected layer -> Fully Connected layer without batches
  //  3) Convolution layer -> Fully Connected layer with batches
  //  4) Fully Connected layer -> Fully Connected layer with batches

  const ITensor *weights_to_use = weights;

  // Check if we have a fully connected layer with batches
  const bool is_batched_fc_layer = output->info()->dimension(1) > 1;
  bool _is_fc_after_conv;
  if (is_batched_fc_layer)
  {
    _is_fc_after_conv =
      (TensorShape::num_max_dimensions >= 4) &&
      (std::equal(input->info()->tensor_shape().cbegin() + 3, input->info()->tensor_shape().cend(),
                  output->info()->tensor_shape().cbegin() + 1));
  }
  else
  {
    _is_fc_after_conv = input->info()->num_dimensions() > 1 && input->info()->dimension(1) > 1;
  }
  ARM_COMPUTE_ERROR_ON_MSG(_is_fc_after_conv,
                           "NEFullyConnectedHybridLayer does not support after conv");
  (void)_is_fc_after_conv;

  // Reshape weights if needed
  if (!_are_weights_reshaped)
  {
    // Reshape the weights
    _reshape_weights_output.allocator()->init(
      weights->info()->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(
        compute_transposed_shape(*weights->info())));
    _reshape_weights_function.configure(weights_to_use, &_reshape_weights_output);
    weights_to_use = &_reshape_weights_output;
  }

  // Quantize input
  _quantized_input.allocator()->init(
    input->info()->clone()->set_is_resizable(true).reset_padding().set_data_type(
      DataType::QASYMM8_SIGNED));
  _scale_factor.allocator()->init(
    TensorInfo(TensorShape{output->info()->dimension(1)}, 1, DataType::F32));
  _quant_input_kernel.configure(input, &_quantized_input, &_scale_factor);

  // GEMM
  _gemmlowp_output.allocator()->init(
    output->info()->clone()->set_is_resizable(true).reset_padding().set_data_type(DataType::S32));
  configure_mm(&_quantized_input, weights_to_use, &_gemmlowp_output);

  // Multiply scale
  _multiply_scale_kernel.configure(&_gemmlowp_output, &_scale_factor, output,
                                   weights->info()->quantization_info().uniform().scale);

  _are_weights_reshaped = _are_weights_reshaped || fc_info.retain_internal_weights;

  _quantized_input.allocator()->allocate();
  _scale_factor.allocator()->allocate();
  _gemmlowp_output.allocator()->allocate();
}

Status NEFullyConnectedHybridLayer::validate(const ITensorInfo *input, const ITensorInfo *weights,
                                             const ITensorInfo *biases, const ITensorInfo *output,
                                             FullyConnectedLayerInfo fc_info)
{
  ARM_COMPUTE_UNUSED(fc_info.retain_internal_weights);
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::F16, DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8_SIGNED);
  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
  ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 2);
  ARM_COMPUTE_RETURN_ERROR_ON(output->num_dimensions() > 2);

  bool weights_reshaped = fc_info.transpose_weights ? fc_info.are_weights_reshaped : true;

  const ITensorInfo &reshaped_weights =
    TensorInfo(weights->clone()->set_is_resizable(true).reset_padding().set_tensor_shape(
      compute_transposed_shape(*weights)));

  // Configure accumulate biases kernel for non quantized asymmetric types
  if (biases != nullptr)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, biases);
    ARM_COMPUTE_RETURN_ON_ERROR(NEGEMMMatrixAccumulateBiasesKernel::validate(output, biases));
  }

  // With the Fully Connected layer we can have 4 different cases:
  //  1) Convolution layer -> Fully Connected layer without batches
  //  2) Fully Connected layer -> Fully Connected layer without batches
  //  3) Convolution layer -> Fully Connected layer with batches
  //  4) Fully Connected layer -> Fully Connected layer with batches

  const ITensorInfo *weights_to_use = weights;

  if (!weights_reshaped)
  {
    // Validate reshape weights kernel
    ARM_COMPUTE_RETURN_ON_ERROR(
      NEFullyConnectedHybridLayerReshapeWeights::validate(weights_to_use, &reshaped_weights));
    weights_to_use = &reshaped_weights;
  }

  // Fully Connected layer after a Fully Connected Layer without batches
  ARM_COMPUTE_RETURN_ERROR_ON(input->dimension(0) != weights_to_use->dimension(1));

  // Validate quantization kernel
  const ITensorInfo &quantized_input = TensorInfo(
    input->clone()->set_is_resizable(true).reset_padding().set_data_type(DataType::QASYMM8_SIGNED));
  const ITensorInfo &scale_factor = TensorInfo(TensorShape{output->dimension(1)}, 1, DataType::F32);
  ARM_COMPUTE_RETURN_ON_ERROR(
    NEQuantizationSymmetricKernel::validate(input, &quantized_input, &scale_factor));

  const ITensorInfo &gemmlowp_output = TensorInfo(
    output->clone()->set_is_resizable(true).reset_padding().set_data_type(DataType::S32));
  // Validate matrix multiply kernel
  ARM_COMPUTE_RETURN_ON_ERROR(validate_mm(quantized_input, *weights_to_use, gemmlowp_output));

  ARM_COMPUTE_RETURN_ON_ERROR(NEMultiplyScaleFactorKernel::validate(
    &gemmlowp_output, &scale_factor, output, weights->quantization_info().uniform().scale));

  return Status{};
}

void NEFullyConnectedHybridLayer::run()
{
  prepare();

  MemoryGroupResourceScope scope_mg(_memory_group);

  // Quantize input
  NEScheduler::get().schedule(&_quant_input_kernel, Window::DimY);

  // Run matrix multiply
  _mm_gemmlowp.run();

  // Multiply scale factor
  NEScheduler::get().schedule(&_multiply_scale_kernel, Window::DimY);

  // Accumulate biases if provided
  if (_accumulate_biases)
  {
    // NEScheduler::get().schedule(&_accumulate_biases_kernel, Window::DimY);
  }
}

void NEFullyConnectedHybridLayer::prepare()
{
  if (!_is_prepared)
  {
    ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

    auto release_unused = [](Tensor *w) {
      if (!w->is_used())
      {
        w->allocator()->free();
      }
    };

    // Reshape of the weights (happens only once)
    if (!_are_weights_reshaped)
    {
      // Run reshape weights kernel and mark weights as unused
      _reshape_weights_output.allocator()->allocate();
      _reshape_weights_function.run();

      _are_weights_reshaped = true;
      // We can not release _original_weights because it can be used in other nodes
    }

    // Prepare GEMM prepare and release unused weights
    _mm_gemmlowp.prepare();

    // Release reshaped weights if unused
    release_unused(&_reshape_weights_output);

    _is_prepared = true;
  }
}
