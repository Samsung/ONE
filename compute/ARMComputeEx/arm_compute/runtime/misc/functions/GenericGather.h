/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file        GenericGather.h
 * @brief       This file contains GenericGather class
 * @ingroup     COM_AI_RUNTIME
 */

#ifndef __ARM_COMPUTE_MISC_GENERIC_GATHER_H__
#define __ARM_COMPUTE_MISC_GENERIC_GATHER_H__

#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/CL/CLTensor.h>

#include <arm_compute/runtime/CL/functions/CLPermute.h>
#include <arm_compute/runtime/CL/functions/CLGatherEx.h>

#include "Utils.h"

namespace arm_compute
{
namespace misc
{

/**
 * @brief Class to run Gather with both CPU and GPU
 */
class GenericGather : public arm_compute::IFunction
{
public:
  GenericGather(void)
  {
    // DO NOTHING
  }

public:
  /**
   * @brief Configure the layer
   * @param[in] input The source tensor
   * @param[in] indices The indices tensor
   * @param[in] output The destination tensor
   * @param[in] axis (Optional) The axis in input to gather indices from
   * @return N/A
   */
  void configure(arm_compute::ITensor *input, arm_compute::ITensor *indices,
                 arm_compute::ITensor *output, int axis = 0);

public:
  /**
   * @brief Run the operation. Must be called after configure().
   * @return N/A
   */
  void run(void) override;

private:
  arm_compute::ITensor *_input{nullptr};
  arm_compute::ITensor *_indices{nullptr};
  arm_compute::ITensor *_output{nullptr};
  int _axis{0};
  arm_compute::CLTensor _cl_permuted;

private:
  arm_compute::CLPermute _cl_permute;
  arm_compute::CLGatherEx _cl_gather;
};

} // namespace misc
} // namespace arm_compute

#endif // __ARM_COMPUTE_MISC_GENERIC_GATHER_H__
