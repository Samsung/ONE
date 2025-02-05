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

/*
 * Copyright (c) 2017 ARM Limited.
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
#include "arm_compute/runtime/CL/functions/CLSplitVEx.h"
#include "support/ToolchainSupport.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/runtime/CL/CLScheduler.h"
#include "src/core/helpers/AutoConfiguration.h"
#include <cassert>

using namespace arm_compute;

namespace
{
Status validate_arguments(const ICLTensor *size_splits, const std::vector<ICLTensor *> &outputs,
                          unsigned int num_splits)
{
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(size_splits->info()->num_dimensions() != 1,
                                  "size_splits must be a 1-D tensor.");
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(num_splits != outputs.size(),
                                  "Number of output tensors does not match number of splits.");
  return Status{};
}

Status validate_slices(const ITensorInfo *input, const std::vector<ITensorInfo *> &outputs,
                       uint32_t split_dim)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input);
  ARM_COMPUTE_RETURN_ERROR_ON(split_dim >= input->num_dimensions());
  ARM_COMPUTE_RETURN_ERROR_ON(outputs.size() < 2);

  // Start/End coordinates
  Coordinates start_coords;
  Coordinates end_coords;
  for (unsigned int d = 0; d < input->num_dimensions(); ++d)
  {
    end_coords.set(d, -1);
  }
  unsigned int axis_offset = 0;
  // Validate output tensors
  for (const auto &output : outputs)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(output);
    // Get output shape
    const TensorShape output_shape = output->tensor_shape();
    ARM_COMPUTE_RETURN_ERROR_ON(output_shape.total_size() == 0);

    const size_t axis_split_step = output_shape[split_dim];

    // Output auto inizialitation if not yet initialized
    TensorInfo tmp_output_info = *output->clone();
    auto_init_if_empty(tmp_output_info,
                       input->clone()->set_is_resizable(true).set_tensor_shape(output_shape));

    // Update coordinate on axis
    start_coords.set(split_dim, axis_offset);
    end_coords.set(split_dim, axis_offset + axis_split_step);

    ARM_COMPUTE_RETURN_ON_ERROR(CLSlice::validate(input, output, start_coords, end_coords));

    axis_offset += axis_split_step;
  }

  return Status{};
}

void configure_slices(const ICLTensor *input, const std::vector<ICLTensor *> &outputs,
                      std::vector<CLSlice> &_slice_functions, uint32_t split_dim)
{
  unsigned int axis_offset = 0;
  // Start/End coordinates
  Coordinates start_coords;
  Coordinates end_coords;
  for (unsigned int d = 0; d < input->info()->num_dimensions(); ++d)
  {
    end_coords.set(d, -1);
  }
  int out_iter = 0;
  for (const auto &output : outputs)
  {
    const TensorShape output_shape = output->info()->tensor_shape();
    auto op_size = output_shape.total_size();
    if (!op_size)
    {
      continue;
    }

    assert(op_size != 0);
    assert(split_dim <= output_shape.num_dimensions());

    const size_t axis_split_step = output_shape[split_dim];

    // Output auto inizialitation if not yet initialized
    TensorInfo tmp_output_info = *output->info()->clone();
    auto_init_if_empty(
      tmp_output_info,
      input->info()->clone()->set_is_resizable(true).set_tensor_shape(output_shape));

    // Update coordinate on axis
    start_coords.set(split_dim, axis_offset);
    end_coords.set(split_dim, axis_offset + axis_split_step);

    // Configure slice function
    _slice_functions[out_iter].configure(input, output, start_coords, end_coords);

    // Set valid region from shape
    outputs[out_iter++]->info()->set_valid_region(ValidRegion(Coordinates(), output_shape));
    axis_offset += axis_split_step;
  }
}

} // namespace

CLSplitVEx::CLSplitVEx()
  : _input(nullptr), _size_splits(nullptr), _outputs(), _num_splits(0), _slice_functions()
{
}

void CLSplitVEx::configure(const ICLTensor *input, const ICLTensor *size_splits, uint32_t split_dim,
                           const std::vector<ICLTensor *> &outputs, unsigned int num_splits)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, size_splits);
  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(size_splits, outputs, num_splits));

  _input = input;
  _size_splits = size_splits;
  _outputs = outputs;
  _num_splits = num_splits;

  // Create tensor slices
  _slice_functions.resize(_num_splits);

  // Extract output tensor info
  std::vector<ITensorInfo *> outputs_info;
  for (auto &&output : _outputs)
  {
    ARM_COMPUTE_ERROR_ON_NULLPTR(output);
    outputs_info.emplace_back(output->info());
  }

  // Validate slices
  ARM_COMPUTE_ERROR_THROW_ON(validate_slices(_input->info(), outputs_info, split_dim));

  // Configure slices
  configure_slices(_input, _outputs, _slice_functions, split_dim);
}

void CLSplitVEx::run()
{
  // execute the slices
  for (unsigned i = 0; i < _outputs.size(); ++i)
  {
    _slice_functions[i].run();
  }
}
