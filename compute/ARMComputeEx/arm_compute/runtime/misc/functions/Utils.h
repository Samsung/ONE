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
 * @file utils.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file contains utils for arm compute library
 */
#ifndef __ARM_COMPUTE_MISC_UTILS_H__
#define __ARM_COMPUTE_MISC_UTILS_H__

#include <string>
#include <cassert>
#include <arm_compute/runtime/CL/CLTensor.h>

#include <arm_compute/core/Coordinates.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Types.h>

// TODO : It should be extracted to independent module.

namespace arm_compute
{
namespace misc
{
namespace utils
{

/**
 * @brief Check if this runtime runs on GPU or NEON
 * @return @c true if GPU mode, otherwise @c false
 */
bool isGpuMode();

#ifndef CAST_CL
#define CAST_CL(tensor) static_cast<::arm_compute::CLTensor *>(tensor)
#endif

#ifndef CAST_NE
#define CAST_NE(tensor) static_cast<::arm_compute::Tensor *>(tensor)
#endif

/**
* @brief      Generate arm compute permutation vector from runtime permutation vector
* @param[in]  rank                 Rank number supported upto 4
* @param[in]  runtime_pv           Integer array for runtime permutation vector
* @return     Permutation vector of arm compute
*/
arm_compute::PermutationVector getARMComputePermutationVector(uint32_t rank,
                                                              const int32_t *runtime_pv);

/**
 * @brief       Set value to arm compute tensor with casting
 * @param[in]   value Value to set
 * @param[out]  to    Target tensor of arm compute
 * @param[in]   id    Position of element
 * @return      N/A
 */
template <typename FromT>
void copyCast(const FromT value, arm_compute::ITensor *to, const arm_compute::Coordinates &id)
{
  switch (to->info()->data_type())
  {
    case arm_compute::DataType::F32:
    {
      *reinterpret_cast<float *>(to->ptr_to_element(id)) = static_cast<float>(value);
      break;
    }
    case arm_compute::DataType::S32:
    {
      *reinterpret_cast<int32_t *>(to->ptr_to_element(id)) = static_cast<int32_t>(value);
      break;
    }
    case arm_compute::DataType::U32:
    {
      *reinterpret_cast<uint32_t *>(to->ptr_to_element(id)) = static_cast<uint32_t>(value);
      break;
    }
    case arm_compute::DataType::QASYMM8:
    {
      float realValue = static_cast<float>(value);
      // NOTE We haven't known the policy of rounding for quantization.
      //      So this is set to a temporary value.
      *(to->ptr_to_element(id)) = quantize_qasymm8(realValue, to->info()->quantization_info(),
                                                   arm_compute::RoundingPolicy::TO_ZERO);
      break;
    }
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

} // namespace utils
} // namespace misc
} // namespace arm_compute

#endif // __ARM_COMPUTE_MISC_UTILS_H__
