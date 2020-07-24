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
 * Copyright (c) 2017-2018 ARM Limited.
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

#include "arm_compute/runtime/CL/functions/CLTransposeConvLayer.h"
#include "arm_compute/core/utils/misc/ShapeCalculatorEx.h"

#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/UtilsEx.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "arm_compute/runtime/CPP/CPPScheduler.h"

#include <memory>
#include <tuple>

using namespace arm_compute;
using namespace arm_compute::misc::shape_calculator;

CLTransposeConvLayer::CLTransposeConvLayer(std::shared_ptr<IMemoryManager> memory_manager) // NOLINT
    : _memory_group(std::move(memory_manager)),
      _scale_f(),
      _conv_f(),
      _flip_weights(),
      _scaled_output(),
      _original_weights(nullptr),
      _weights_flipped(),
      _is_prepared(false)
{
}

Status CLTransposeConvLayer::validate(const ITensorInfo *input, const ITensorInfo *weights,
                                      const ITensorInfo *bias, ITensorInfo *output,
                                      const PadStrideInfo &info, unsigned int invalid_right,
                                      unsigned int invalid_bottom, const WeightsInfo &weights_info)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, weights, output);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::QASYMM8, DataType::F16,
                                                       DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, weights);

  const DataLayout data_layout = input->data_layout();

  const size_t idx_w = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
  const size_t idx_h = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
  const size_t idx_c = get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL);

  ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_w) != weights->dimension(idx_h));
  ARM_COMPUTE_RETURN_ERROR_ON(weights->dimension(idx_w) < 1);

  const unsigned int kernel_x = weights->dimension(idx_w);
  const unsigned int kernel_y = weights->dimension(idx_h);

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(invalid_right > kernel_x - 1,
                                  "invalid_right must be smaller than kernel_x");
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(invalid_bottom > kernel_y - 1,
                                  "inner_border_top must be smaller than kernel_y");

  // NOTE From the existing CLDeconvolutionLayer, invalid_right and invalid_bottom were added.
  auto out_dims = transposeconv_output_dimensions(
      input->dimension(idx_w), input->dimension(idx_h), weights->dimension(idx_w),
      weights->dimension(idx_h), info, invalid_right, invalid_bottom);

  const TensorShape output_shape = compute_transposeconv_output_shape(out_dims, *input, *weights);

  ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output, weights);

  if (bias != nullptr)
  {
    if (is_data_type_quantized_asymmetric(input->data_type()))
    {
      ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);
    }
    else
    {
      ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, bias);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, bias);
  }

  ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(idx_w) != output_shape[idx_w],
                                  "Output's width is invalid.");
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(idx_h) != output_shape[idx_h],
                                  "Output's height is invalid.");
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(output->dimension(idx_c) != output_shape[idx_c],
                                  "Output's depth is invalid.");

  unsigned int pad_left = 0;
  unsigned int pad_right = 0;
  unsigned int pad_top = 0;
  unsigned int pad_bottom = 0;
  const TensorShape scale_out_shape = compute_transposeconv_upsampled_shape(
      *input, *weights, info, out_dims, invalid_right, invalid_bottom, pad_left, pad_right, pad_top,
      pad_bottom);
  TensorInfo scale_out_info(input->clone()
                                ->set_is_resizable(true)
                                .reset_padding()
                                .set_tensor_shape(scale_out_shape)
                                .set_data_layout(data_layout));
  const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);

  ARM_COMPUTE_RETURN_ON_ERROR(
      CLTransposeConvLayerUpsample::validate(input, &scale_out_info, BorderSize(0, 0), info));
  ARM_COMPUTE_RETURN_ON_ERROR(CLConvolutionLayer::validate(&scale_out_info, weights, bias, output,
                                                           conv_info, weights_info));

  return Status{};
}

void CLTransposeConvLayer::configure(ICLTensor *input, ICLTensor *weights, const ICLTensor *bias,
                                     ICLTensor *output, const PadStrideInfo &info,
                                     unsigned int invalid_right, unsigned int invalid_bottom,
                                     const WeightsInfo &weights_info)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);

  const unsigned int stride_x = info.stride().first;
  const unsigned int stride_y = info.stride().second;

  const DataLayout data_layout = input->info()->data_layout();

  const size_t idx_w = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
  const size_t idx_h = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);

  _original_weights = weights;
  _weights_flipped.allocator()->init(weights->info()->clone()->set_data_layout(data_layout));
  _flip_weights.configure(weights, &_weights_flipped);

  // NOTE From the existing CLDeconvolutionLayer, invalid_right and invalid_bottom were
  // added.
  auto out_dims = transposeconv_output_dimensions(
      input->info()->dimension(idx_w), input->info()->dimension(idx_h),
      weights->info()->dimension(idx_w), weights->info()->dimension(idx_h), info, invalid_right,
      invalid_bottom);

  const TensorShape output_shape =
      compute_transposeconv_output_shape(out_dims, *input->info(), *weights->info());

  // Output auto initialization if not yet initialized
  auto_init_if_empty(
      *output->info(),
      input->info()->clone()->set_tensor_shape(output_shape).set_data_layout(data_layout));

  // Perform validation step
  ARM_COMPUTE_ERROR_THROW_ON(CLTransposeConvLayer::validate(
      input->info(), weights->info(), bias == nullptr ? nullptr : bias->info(), output->info(),
      info, invalid_right, invalid_bottom));

  _is_prepared = weights_info.retain_internal_weights();

  _memory_group.manage(&_scaled_output);

  // Find the upsampled dimensions and the padding needed for the convolution with stride 1 in order
  // to match output shape
  unsigned int pad_left = 0;
  unsigned int pad_right = 0;
  unsigned int pad_top = 0;
  unsigned int pad_bottom = 0;
  const TensorShape scale_out_shape = compute_transposeconv_upsampled_shape(
      *input->info(), *weights->info(), info, out_dims, invalid_right, invalid_bottom, pad_left,
      pad_right, pad_top, pad_bottom);

  TensorInfo scale_out_info(scale_out_shape, 1, input->info()->data_type(),
                            input->info()->quantization_info());
  scale_out_info.set_data_layout(data_layout);
  _scaled_output.allocator()->init(scale_out_info);

  // configure scale function
  const PadStrideInfo upsample_info(stride_x, stride_y, pad_left, pad_right, pad_top, pad_bottom,
                                    DimensionRoundingType::FLOOR);
  _scale_f.configure(input, &_scaled_output, BorderSize(0, 0), upsample_info);

  // setup the function to convolve the upscaled output
  const PadStrideInfo conv_info(1, 1, 0, 0, 0, 0, DimensionRoundingType::CEIL);
  _conv_f.configure(&_scaled_output, &_weights_flipped, bias, output, conv_info, weights_info);
  _scaled_output.allocator()->allocate();
}

void CLTransposeConvLayer::run()
{
  prepare();

  _memory_group.acquire();

  _scale_f.run();
  _conv_f.run();

  _memory_group.release();
}

void CLTransposeConvLayer::prepare()
{
  if (!_is_prepared)
  {
    ARM_COMPUTE_ERROR_ON(!_original_weights->is_used());

    // Run weights flipping and mark original weights tensor as unused
    _weights_flipped.allocator()->allocate();
    _weights_flipped.map(true);
    _original_weights->map(CLScheduler::get().queue(), true);
    CPPScheduler::get().schedule(&_flip_weights, Window::DimZ);
    _weights_flipped.unmap();
    _original_weights->unmap(CLScheduler::get().queue());
    _original_weights->mark_as_unused();

    // Prepare convolution
    _conv_f.prepare();

    if (!_weights_flipped.is_used())
    {
      _weights_flipped.allocator()->free();
    }

    _is_prepared = true;
  }
}
