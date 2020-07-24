/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_TENSOR_UTILS_H__
#define __NNFW_CKER_TENSOR_UTILS_H__

#include "cker/Types.h"
#include "cker/PortableTensorUtils.h"
#include "cker/NeonTensorUtils.h"
#include "cker/neon/neon_check.h"

#include <cstring>
#include <cmath>

namespace nnfw
{
namespace cker
{

void VectorBatchVectorAssign(const float *vector, int v_size, int n_batch, float *batch_vector)
{
  PortableVectorBatchVectorAssign(vector, v_size, n_batch, batch_vector);
}

bool IsZeroVector(const float *vector, int v_size)
{
  return NEON_OR_PORTABLE(IsZeroVector, vector, v_size);
}

void ApplyActivationToVector(const float *vector, int v_size,
                             FusedActivationFunctionType activation, float *result)
{
  PortableApplyActivationToVector(vector, v_size, activation, result);
}

void SymmetricQuantizeFloats(const float *values, const int size, int8_t *quantized_values,
                             float *min, float *max, float *scaling_factor)
{
  return NEON_OR_PORTABLE(SymmetricQuantizeFloats, values, size, quantized_values, min, max,
                          scaling_factor);
}

void MatrixBatchVectorMultiplyAccumulate(const int8_t *matrix, const int m_rows, const int m_cols,
                                         const int8_t *vector, const float *scaling_factors,
                                         int n_batch, float *result, int result_stride)
{
  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols, vector,
                   scaling_factors, n_batch, result, result_stride);
}

void MatrixBatchVectorMultiplyAccumulate(const float *matrix, int m_rows, int m_cols,
                                         const float *vector, int n_batch, float *result,
                                         int result_stride)
{
  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols, vector, n_batch,
                   result, result_stride);
}

void MatrixBatchVectorMultiplyAccumulate(const int8_t *matrix, const int m_rows, const int m_cols,
                                         const int8_t *vectors, const float *scaling_factors,
                                         int n_batch, int32_t *scratch, float *result,
                                         int result_stride, ruy::Context *ruy_context)
{
  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols, vectors,
                   scaling_factors, n_batch, scratch, result, result_stride, ruy_context);
}

void ZeroVector(float *vector, int v_size) { PortableZeroVector(vector, v_size); }

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TENSOR_UTILS_H__
