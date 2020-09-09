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

#include "arm_compute/runtime/CL/functions/CLReduceOperation.h"

#include "arm_compute/core/CL/kernels/CLReduceOperationKernel.h"
#include "arm_compute/core/TensorShape.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute;

CLReduceOperation::CLReduceOperation(std::shared_ptr<IMemoryManager> memory_manager)
    : _memory_group(std::move(memory_manager)), _input(nullptr), _output(nullptr), _axis(),
      _keep_dims(false), _interm_tensors(), _reduce_kernels(), _reshape()
{
}

Status CLReduceOperation::validate(const ITensorInfo *input, const ITensorInfo *output,
                                   const std::set<uint32_t> &axis, bool keep_dims,
                                   const ReduceOperation &op)
{
  const size_t num_of_kernels = axis.size();
  const size_t num_of_interm_tensors = num_of_kernels - (keep_dims ? 1 : 0);

  ARM_COMPUTE_RETURN_ERROR_ON(num_of_kernels < 1);

  // Create temporary tensor infos
  auto interm_tensors = support::cpp14::make_unique<TensorInfo[]>(num_of_interm_tensors);

  // Create intermediate tensor info
  TensorShape shape{input->tensor_shape()};

  auto it = axis.begin();
  for (size_t i = 0; i < num_of_interm_tensors; ++i, ++it)
  {
    shape.set(*it, 1, false);
    interm_tensors[i].set_data_type(input->data_type());
    interm_tensors[i].set_tensor_shape(shape);
    interm_tensors[i].set_num_channels(input->num_channels());
    interm_tensors[i].set_data_layout(input->data_layout());
    interm_tensors[i].set_quantization_info(input->quantization_info());
  }

  // Set a vector that is ordered ITensorInfo sequentially.
  std::vector<const ITensorInfo *> tensors;
  tensors.emplace_back(input);
  for (size_t i = 0; i < num_of_interm_tensors; ++i)
  {
    tensors.emplace_back(interm_tensors.get() + i);
  }
  tensors.emplace_back(output);

  // Validate ReduceOperation only on all kernels
  it = axis.begin();
  for (size_t i = 0; i < num_of_kernels; ++i, ++it)
  {
    ARM_COMPUTE_RETURN_ON_ERROR(
        CLReduceOperationKernel::validate(tensors[i], tensors[i + 1], *it, op));
  }

  if (!keep_dims)
  {
    ARM_COMPUTE_RETURN_ON_ERROR(
        CLReshapeLayer::validate(&interm_tensors[num_of_interm_tensors - 1], output));
  }

  return Status{};
}

void CLReduceOperation::configure(ICLTensor *input, ICLTensor *output,
                                  const std::set<uint32_t> &axis, bool keep_dims,
                                  ReduceOperation op)
{
  ARM_COMPUTE_ERROR_THROW_ON(validate(input->info(), output->info(), axis, keep_dims, op));

  _axis = axis;

  _input = input;
  _output = output;
  _keep_dims = keep_dims;

  // NOTE The axis must have no duplication.
  const size_t num_of_kernels = axis.size();
  const size_t num_of_interm_tensors = num_of_kernels - (keep_dims ? 1 : 0);

  if (num_of_kernels < 1)
  {
    throw std::runtime_error("CLReduceOperation: there is no axis to reduce");
  }

  _interm_tensors = support::cpp14::make_unique<CLTensor[]>(num_of_interm_tensors);
  _reduce_kernels = support::cpp14::make_unique<CLReduceOperationKernel[]>(num_of_kernels);

  // Set a vector that is ordered ICLTensors sequentially.
  std::vector<ICLTensor *> tensors;
  tensors.emplace_back(input);
  for (size_t i = 0; i < num_of_interm_tensors; ++i)
  {
    tensors.emplace_back(_interm_tensors.get() + i);
  }
  tensors.emplace_back(output);

  // Apply ReduceOperation on all kernels
  TensorShape shape{input->info()->tensor_shape()};
  auto it = axis.begin();
  for (size_t i = 0; i < num_of_kernels; ++i, ++it)
  {
    shape.set(*it, 1, false);
    if (!keep_dims || i != (num_of_kernels - 1))
    {
      _interm_tensors[i].allocator()->init(input->info()->clone()->set_tensor_shape(shape));
      _memory_group.manage(&_interm_tensors[i]);
    }
    _reduce_kernels[i].configure(tensors[i], tensors[i + 1], *it, op);
    if (i != 0)
    {
      _interm_tensors[i - 1].allocator()->allocate();
    }
  }

  // Configure reshape layer if we want to drop the dimensions
  if (!keep_dims)
  {
    _reshape.configure(&_interm_tensors[num_of_interm_tensors - 1], output);
    _interm_tensors[num_of_interm_tensors - 1].allocator()->allocate();
  }
}

void CLReduceOperation::run()
{
  MemoryGroupResourceScope scope_mg(_memory_group);

  const size_t num_of_kernels = _axis.size();
  for (size_t i = 0; i < num_of_kernels; ++i)
  {
    CLScheduler::get().enqueue(_reduce_kernels[i]);
  }

  if (!_keep_dims)
  {
    _reshape.run();
  }
}
