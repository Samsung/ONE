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
 * Copyright (c) 2016-2018 ARM Limited.
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

/**
 * @file CLReduceOperation.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains arm_compute::CLReduceOperation class
 */

#ifndef __ARM_COMPUTE_CLREDUCEOPERATION_H__
#define __ARM_COMPUTE_CLREDUCEOPERATION_H__

#include "arm_compute/core/CL/kernels/CLReduceOperationKernel.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/MemoryGroup.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#include "arm_compute/runtime/CL/CLTensorAllocator.h"
#include "arm_compute/runtime/CL/functions/CLReshapeLayer.h"
#include "arm_compute/runtime/IFunction.h"
#include "arm_compute/runtime/IMemoryManager.h"

namespace arm_compute
{
class ICLTensor;

/**
 * @brief Class to perform ReduceOperation
 */
class CLReduceOperation : public IFunction
{
public:
  /**
   * @brief Construct a new ReduceOperation object
   */
  CLReduceOperation(std::shared_ptr<IMemoryManager> memory_manager);

  /**
   * @brief Set the input and output tensors.
   * @param[in]  input     Source tensor. Data types supported: U8/S32/F32
   * @param[out] output    Destination tensor. Data types and data layouts supported: Same as @p
   * input.
   * @param[in]  axis      Axis along which to reduce. It must be sorted and no duplicates.
   * @param[in]  keep_dims If positive, retains reduced dimensions with length 1.
   * @param[in]  op        Reduce operation to perform.
   * @return N/A
   */
  void configure(ICLTensor *input, ICLTensor *output, const std::set<uint32_t> &axis,
                 bool keep_dims, ReductionOperation op);

  /**
   * @brief Static function to check if given info will lead to a valid configuration of @ref
   *        CLReduceOperation.
   * @param[in] input     Source tensor info. Data types supported: U8/S32/F32
   * @param[in] output    Destination tensor info. Data types and data layouts supported: Same as @p
   * input.
   * @param[in] axis      Axis along which to reduce. It must be sorted and no duplicates.
   * @param[in] keep_dims If positive, retains reduced dimensions with length 1.
   * @param[in] op        Reduce operation to perform.
   * @return a status
   */
  static Status validate(const ITensorInfo *input, const ITensorInfo *output,
                         const std::set<uint32_t> &axis, bool keep_dims,
                         const ReductionOperation &op);

  /**
   * @brief Run the OpenCL kernel for this operation
   * @return N/A
   */
  void run() override;

private:
  MemoryGroup _memory_group;
  ICLTensor *_input;
  ICLTensor *_output;
  std::set<uint32_t> _axis;
  bool _keep_dims;

  std::unique_ptr<CLTensor[]> _interm_tensors{nullptr};
  std::unique_ptr<CLReduceOperationKernel[]> _reduce_kernels{nullptr};
  CLReshapeLayer _reshape;
};
}
#endif /*__ARM_COMPUTE_CLREDUCEOPERATION_H__ */
