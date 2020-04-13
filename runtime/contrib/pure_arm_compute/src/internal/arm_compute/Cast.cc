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

#include "internal/arm_compute/Cast.h"

#include "internal/Swizzle.h"

::arm_compute::Coordinates getARMComputeAxises(uint32_t rank)
{
  ::arm_compute::Coordinates res{};

  res.set_num_dimensions(rank);

  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    res.set(axis, ToARMComputeAxis(rank, axis).value());
  }

  return res;
}

::arm_compute::Coordinates asARMComputeCoordinates(const ::arm_compute::Coordinates &runtime_coord,
                                                   const ::arm_compute::Coordinates &axises)
{
  ::arm_compute::Coordinates id{};
  assert(runtime_coord.num_dimensions() == axises.num_dimensions());
  for (size_t i = 0; i < runtime_coord.num_dimensions(); ++i)
  {
    id.set(axises[i], runtime_coord[i]);
  }
  return id;
}

// Restructure runtime_permutationVector to ACL_permutationVector
::arm_compute::PermutationVector getARMComputePermutationVector(uint32_t rank,
                                                                const int32_t *runtime_pv)
{
  // rank upto 4 is supported
  assert(rank <= 4);
  assert(runtime_pv != nullptr);

  int new_pv[4] = {0};
  ::arm_compute::Coordinates axises = getARMComputeAxises(rank);

  for (uint32_t i = 0; i < rank; ++i)
  {
    new_pv[axises[i]] = ToARMComputeAxis(rank, runtime_pv[i]).value();
  }

  ::arm_compute::PermutationVector ACL_PV =
      ::arm_compute::PermutationVector{new_pv[0], new_pv[1], new_pv[2], new_pv[3]};
  ACL_PV.set_num_dimensions(rank);

  return ACL_PV;
}

::arm_compute::TensorShape asTensorShape(const internal::tflite::operand::Shape &shape,
                                         bool apply_dim_correction)
{
  const uint32_t rank = shape.rank();

  ::arm_compute::TensorShape res{};

  res.set_num_dimensions(rank);

  for (uint32_t axis = 0; axis < rank; ++axis)
  {
    // NOTE In some cases, in incorrect dimensions is required.
    // For example, intput_size is 1 in LSTM. The input-to-input weights([num_units, input_size]) of
    // LSTM is used as the weight of the FullyConnected.
    // The FullyConnected's weight must be greater or equal than 2-dimensions.
    // However, if the dimension correction is applied to input_to_input_weights with input_size
    // equal to 1, it will be changed to 1-D.
    // So input_to_input_weights is not used by the weight of FullyConnected.
    res.set(ToARMComputeAxis(rank, axis).value(), shape.dim(axis), apply_dim_correction);
  }

  return res;
}

::arm_compute::DataType asDataType(const int32_t type)
{
  switch (type)
  {
    case ANEURALNETWORKS_FLOAT32:
    case ANEURALNETWORKS_TENSOR_FLOAT32:
      return ::arm_compute::DataType::F32;
    case ANEURALNETWORKS_INT32:
    case ANEURALNETWORKS_TENSOR_INT32:
      return ::arm_compute::DataType::S32;
    case ANEURALNETWORKS_UINT32:
      return ::arm_compute::DataType::U32;
    case ANEURALNETWORKS_TENSOR_QUANT8_ASYMM:
      return ::arm_compute::DataType::QASYMM8;
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

::arm_compute::ActivationLayerInfo asActivationInfo(FuseCode code)
{
  switch (code)
  {
    case ANEURALNETWORKS_FUSED_NONE:
      return ::arm_compute::ActivationLayerInfo{};
    case ANEURALNETWORKS_FUSED_RELU:
      return ::arm_compute::ActivationLayerInfo{
          ::arm_compute::ActivationLayerInfo::ActivationFunction::RELU};
    case ANEURALNETWORKS_FUSED_RELU1:
      return ::arm_compute::ActivationLayerInfo{
          ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 1.0f, -1.0f};
    case ANEURALNETWORKS_FUSED_RELU6:
      return ::arm_compute::ActivationLayerInfo{
          ::arm_compute::ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU, 6.0f, 0.0f};
    default:
      throw std::runtime_error("Not supported, yet");
      break;
  }
}

::arm_compute::QuantizationInfo asQuantizationInfo(const float scale, const int32_t offset)
{
  return ::arm_compute::QuantizationInfo(scale, offset);
}

::arm_compute::TensorInfo asTensorInfo(const ::arm_compute::TensorShape &shape, const int32_t type,
                                       const float scale, const int32_t zeroPoint)
{
  return ::arm_compute::TensorInfo(shape, 1, asDataType(type),
                                   asQuantizationInfo(scale, zeroPoint));
}

::arm_compute::TensorInfo asTensorInfo(const ::arm_compute::TensorShape &shape,
                                       const ::arm_compute::DataType &type, const float scale,
                                       const int32_t zeroPoint)
{
  return ::arm_compute::TensorInfo(shape, 1, type, asQuantizationInfo(scale, zeroPoint));
}
