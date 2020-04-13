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
 * @file     TensorShapeUtils.h
 * @brief    This file contains utilities function of tensor shape
 * @ingroup  COM_AI_RUNTIME
 */

#ifndef __NNFW_TFLITE_TENSOR_SHAPE_UTILS_H__
#define __NNFW_TFLITE_TENSOR_SHAPE_UTILS_H__

#include "misc/tensor/Shape.h"

#include <vector>

namespace nnfw
{
namespace tflite
{

/**
 * @brief Converts tensor::Shape into a vector
 * @param[in] shape The tensor shape to be converted
 * @return vector value of given shape object
 */
static inline std::vector<int32_t> as_dims(const nnfw::misc::tensor::Shape &shape)
{
  std::vector<int32_t> dims;

  for (uint32_t axis = 0; axis < shape.rank(); ++axis)
  {
    dims.emplace_back(shape.dim(axis));
  }

  return dims;
}

/**
 * @brief Broadcasts between two given shapes
 * @param[in] lhs_shape The left hand side shape
 * @param[in] rhs_shape The right hand side shape
 * @return The broadcasted shape
 */
nnfw::misc::tensor::Shape broadcast(const nnfw::misc::tensor::Shape &lhs_shape,
                                    const nnfw::misc::tensor::Shape &rhs_shape);

} // namespace tflite
} // namespace nnfw

#endif // __NNFW_TFLITE_TENSOR_SHAPE_UTILS_H__
