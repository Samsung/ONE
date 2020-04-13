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

#include "arm_compute/runtime/CL/functions/CLArgOperation.h"

#include "arm_compute/core/CL/kernels/CLArgOperationKernel.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

namespace arm_compute
{

CLArgOperation::CLArgOperation()
{
  // DO NOTHING
}

void CLArgOperation::configure(ICLTensor *input, ICLTensor *output, std::vector<uint32_t> axis,
                               ArgOperation op)
{
  ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), axis, output->info(), op));
  _input = input;
  _output = output;
  _axis = axis;
  _arg_op = op;
  // NOTE The argminmax_axis must have no duplication.
  _num_of_kernels = axis.size();
  const size_t num_of_interm_tensors = _num_of_kernels - 1;

  _interm_tensors = arm_compute::support::cpp14::make_unique<CLTensor[]>(num_of_interm_tensors);
  _argop_kernels =
      arm_compute::support::cpp14::make_unique<CLArgOperationKernel[]>(_num_of_kernels);

  TensorShape shape{input->info()->tensor_shape()};
  for (size_t i = 0; i < num_of_interm_tensors; i++)
  {
    shape.set(_axis[i], 1);
    _interm_tensors[i].allocator()->init(
        TensorInfo(shape, input->info()->num_channels(), input->info()->data_type())
            .set_data_layout(input->info()->data_layout()));
    _interm_tensors[i].allocator()->allocate();
  }

  // Set a vector that is ordered ICLTensors sequentially.
  std::vector<ICLTensor *> tensors;
  tensors.emplace_back(input);
  for (size_t i = 0; i < num_of_interm_tensors; i++)
  {
    tensors.emplace_back(_interm_tensors.get() + i);
  }
  tensors.emplace_back(output);

  // Apply ArgMinMax on all kernels
  for (size_t i = 0; i < _num_of_kernels; i++)
  {
    _argop_kernels[i].configure(tensors[i], tensors[i + 1], _axis[i], op);
  }
}

Status CLArgOperation::validate(const ITensorInfo *input, const std::vector<uint32_t> &axis,
                                const ITensorInfo *output, ArgOperation op)
{
  const size_t num_of_kernels = axis.size();
  const size_t num_of_interm_tensors = num_of_kernels - 1;

  // Create temporary tensor infos
  auto interm_tensors =
      arm_compute::support::cpp14::make_unique<TensorInfo[]>(num_of_interm_tensors);

  // Create intermediate tensor info
  TensorShape shape{input->tensor_shape()};

  for (size_t i = 0; i < num_of_interm_tensors; i++)
  {
    shape.set(axis[i], 1);
    interm_tensors[i].set_data_type(input->data_type());
    interm_tensors[i].set_tensor_shape(shape);
    interm_tensors[i].set_num_channels(input->num_channels());
  }

  // Set a vector that is ordered ITensorInfo sequentially.
  std::vector<const ITensorInfo *> tensors;
  tensors.emplace_back(input);
  for (size_t i = 0; i < num_of_interm_tensors; i++)
  {
    tensors.emplace_back(interm_tensors.get() + i);
  }
  tensors.emplace_back(output);

  // Validate argminmax only on all kernels
  for (size_t i = 0; i < num_of_kernels; i++)
  {
    ARM_COMPUTE_RETURN_ON_ERROR(
        CLArgOperationKernel::validate(tensors[i], tensors[i + 1], axis[i], op));
  }

  return Status{};
}

void CLArgOperation::run()
{
  for (size_t i = 0; i < _num_of_kernels; ++i)
  {
    CLScheduler::get().enqueue(_argop_kernels[i]);
  }
}

} // namespace arm_compute
