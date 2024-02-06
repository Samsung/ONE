/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_GPU_CL_TENSOR_BUILDER_HELPER_H__
#define __ONERT_BACKEND_GPU_CL_TENSOR_BUILDER_HELPER_H__

#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"

#include "ir/operation/BinaryArithmetic.h"
#include "ir/operation/ElementwiseActivation.h"
#include "ir/operation/ElementwiseBinary.h"
#include "ir/operation/ElementwiseUnary.h"
#include "ir/operation/Pool2D.h"

namespace onert
{
namespace backend
{
namespace gpu_cl
{

inline tflite::gpu::HW ToHW(int32_t h, int32_t w)
{
  return tflite::gpu::HW(h > 0 ? h : 1, w > 0 ? w : 1);
}

template <typename AttrT>
inline void UpdatePadding(const ir::PaddingType type, const tflite::gpu::BHWC &input_shape,
                          AttrT *attr)
{
  if (type == ir::PaddingType::SAME)
  {
    attr->padding = CalculateSamePadding(input_shape, *attr);
  }
  else
  {
    attr->padding.prepended = tflite::gpu::HW(0, 0);
    attr->padding.appended = tflite::gpu::HW(0, 0);
  }
}

inline tflite::gpu::PoolingType convertPoolType(ir::operation::Pool2D::PoolType type_ir)
{
  switch (type_ir)
  {
    case ir::operation::Pool2D::PoolType::AVG:
      return tflite::gpu::PoolingType::AVERAGE;
    case ir::operation::Pool2D::PoolType::MAX:
      return tflite::gpu::PoolingType::MAX;
    default:
      throw std::runtime_error("gpu_Cl KernelGenerator : Not supported operation yet");
  }
}

inline tflite::gpu::BHWC ToBHWC(ir::Shape shape)
{
  switch (shape.rank())
  {
    case 1:
      // B layout
      return tflite::gpu::BHWC(shape.dim(0), 1, 1, 1);
      break;
    case 2:
      // BC layout
      return tflite::gpu::BHWC(shape.dim(0), 1, 1, shape.dim(1));
      break;
    case 3:
      // BWC layout
      return tflite::gpu::BHWC(shape.dim(0), 1, shape.dim(1), shape.dim(2));
      break;
    case 4:
      // BHWC layout
      return tflite::gpu::BHWC(shape.dim(0), shape.dim(1), shape.dim(2), shape.dim(3));
      break;
    default:
      break;
  }
  return tflite::gpu::BHWC();
}

inline bool CheckIfLinearConvertible(const ir::Shape *shape)
{
  if (shape->num_elements() <= 0)
  {
    return false;
  }
  for (int i = 0; i < shape->rank() - 1; ++i)
  {
    if (shape->dim(i) != 1)
    {
      return false;
    }
  }
  return true;
}

inline tflite::gpu::OperationType
convertArithmeticType(ir::operation::BinaryArithmetic::ArithmeticType arithmetic_type_ir)
{
  switch (arithmetic_type_ir)
  {
    case ir::operation::BinaryArithmetic::ArithmeticType::ADD:
      return tflite::gpu::OperationType::ADD;
    case ir::operation::BinaryArithmetic::ArithmeticType::SUB:
      return tflite::gpu::OperationType::SUB;
    case ir::operation::BinaryArithmetic::ArithmeticType::MUL:
      return tflite::gpu::OperationType::MUL;
    case ir::operation::BinaryArithmetic::ArithmeticType::DIV:
      return tflite::gpu::OperationType::DIV;
    default:
      throw std::runtime_error("Unsupported ArithmeticType");
  }
}

inline tflite::gpu::OperationType
convertElementwiseActivationType(ir::operation::ElementwiseActivation::Type type_ir)
{
  switch (type_ir)
  {
    case ir::operation::ElementwiseActivation::Type::LOGISTIC:
      return tflite::gpu::OperationType::SIGMOID;
    default:
      throw std::runtime_error("Unsupported ElementwiseActivationType");
  }
}

enum TensorType
{
  TENSOR_TYPE_VALID = 0,
  TENSOR_TYPE_INPUT = 1,
  TENSOR_TYPE_OUTPUT = 2,
  TENSOR_TYPE_DELETE = 3
};

} // namespace gpu_cl
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_GPU_CL_TENSOR_BUILDER_HELPER_H__
