/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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
 * @file    Cast.h
 * @ingroup COM_AI_RUNTIME
 * @brief   This file defines casting functions from internal object to arm compute object
 */
#ifndef __ARM_COMPUTE_CAST_H__
#define __ARM_COMPUTE_CAST_H__

#include <arm_compute/core/Coordinates.h>
#include <arm_compute/core/TensorInfo.h>
#include <arm_compute/core/TensorShape.h>
#include <arm_compute/core/Types.h>

#include <NeuralNetworks.h>

#include "internal/Model.h"

/**
 * @brief     Generate arm compute coordinate object from rank
 * @param[in] rank  Rank number
 * @return    Coordinate object
 */
::arm_compute::Coordinates getARMComputeAxises(uint32_t rank);

/**
 * @brief     Generate arm compute coordinate object from runtime coordinate object
 * @param[in] runtime_coord        Runtime coordinates object
 * @param[in] axises               Coordinates for axises to map runtime-coordinates to
 *                                 arm_compute-coordinates
 * @return    Arm_compute coordinate object
 */
::arm_compute::Coordinates asARMComputeCoordinates(const ::arm_compute::Coordinates &runtime_coord,
                                                   const ::arm_compute::Coordinates &axises);

/**
* @brief      Generate arm compute permutation vector from runtime permutation vector
* @param[in]  rank                 Rank number supported upto 4
* @param[in]  runtime_pv           Integer array for runtime permutation vector
* @return     Permutation vector of arm compute
*/
::arm_compute::PermutationVector getARMComputePermutationVector(uint32_t rank,
                                                                const int32_t *runtime_pv);
/**
 * @brief     Cast from shape of internal to arm compute
 * @param[in] shape                 Internal shape object
 * @param[in] apply_dim_correction  Flag to state whether apply dimension correction after setting
 *                                  one dimension in arm compute
 * @return    TensorShape object of arm compute
 */
::arm_compute::TensorShape asTensorShape(const internal::tflite::operand::Shape &shape,
                                         bool apply_dim_correction = true);

/**
 * @brief     Cast from data type enum of NNAPI to arm compute
 * @param[in] type  NNAPI data type
 * @return    Data type of arm compute
 */
::arm_compute::DataType asDataType(const int32_t type);

/**
 * @brief     Cast from NNAPI activation type enum to activation object of arm compute
 * @param[in] code  NNAPI activation type
 * @return    ActivationLayerInfo object of arm compute
 */
::arm_compute::ActivationLayerInfo asActivationInfo(FuseCode code);

/**
 * @brief     Generate quantization info object of arm compute
 * @param[in] scale   Scale of quantization
 * @param[in] offset  Offset of quantization
 * @return    QuantizationInfo object of arm compute
 */
::arm_compute::QuantizationInfo asQuantizationInfo(const float scale, const int32_t offset);

/**
 * @brief     Cast from internal tensor info to tensor info object of arm compute
 * @param[in] shape     Tensor shape
 * @param[in] type      Tensor type
 * @param[in] scale     Scale of tensor quantization
 * @param[in] zeroPoint Zeropoint of tensor quantization
 * @return    TensorInfo object of arm compute
 */
::arm_compute::TensorInfo asTensorInfo(const ::arm_compute::TensorShape &shape, const int32_t type,
                                       const float scale = 0.0f, const int32_t zeroPoint = 0);

/**
 * @brief     Cast from internal tensor info to tensor info object of arm compute
 * @param[in] shape     Tensor shape
 * @param[in] type      Tensor type of arm compute
 * @param[in] scale     Scale of tensor quantization
 * @param[in] zeroPoint Zeropoint of tensor quantization
 * @return    TensorInfo object of arm compute
 */
::arm_compute::TensorInfo asTensorInfo(const ::arm_compute::TensorShape &shape,
                                       const ::arm_compute::DataType &type, const float scale,
                                       const int32_t zeroPoint);

/**
 * @brief       Set value to arm compute tensor with casting
 * @param[in]   value Value to set
 * @param[out]  to    Target tensor of arm compute
 * @param[in]   id    Position of element
 * @return      N/A
 */
template <typename FromT>
void copyCast(const FromT value, ::arm_compute::ITensor *to, const ::arm_compute::Coordinates &id)
{
  switch (to->info()->data_type())
  {
    case ::arm_compute::DataType::F32:
    {
      *reinterpret_cast<float *>(to->ptr_to_element(id)) = static_cast<float>(value);
      break;
    }
    case ::arm_compute::DataType::S32:
    {
      *reinterpret_cast<int32_t *>(to->ptr_to_element(id)) = static_cast<int32_t>(value);
      break;
    }
    case ::arm_compute::DataType::U32:
    {
      *reinterpret_cast<uint32_t *>(to->ptr_to_element(id)) = static_cast<uint32_t>(value);
      break;
    }
    case ::arm_compute::DataType::QASYMM8:
    {
      float realValue = static_cast<float>(value);
      // NOTE We haven't known the policy of rounding for quantization.
      //      So this is set to a temporary value.
      *(to->ptr_to_element(id)) = to->info()->quantization_info().quantize(
          realValue, ::arm_compute::RoundingPolicy::TO_ZERO);
      break;
    }
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

#endif // __ARM_COMPUTE_CAST_H__
