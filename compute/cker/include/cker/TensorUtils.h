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

inline void CwiseClipping(float *vector, const int v_size, const float clipping_value)
{
  NEON_OR_PORTABLE(CwiseClipping, vector, v_size, clipping_value);
}

inline void VectorBatchVectorAdd(const float *vector, int v_size, int n_batch, float *batch_vector)
{
  PortableVectorBatchVectorAdd(vector, v_size, n_batch, batch_vector);
}

inline void VectorBatchVectorAssign(const float *vector, int v_size, int n_batch,
                                    float *batch_vector)
{
  PortableVectorBatchVectorAssign(vector, v_size, n_batch, batch_vector);
}

// Cwise product of two vectors.
template <typename T>
inline void VectorVectorCwiseProduct(const T *__restrict__ vector1, const T *__restrict__ vector2,
                                     int v_size, T *__restrict__ result)
{
  for (int v = 0; v < v_size; v++)
  {
    *result++ = *vector1++ * *vector2++;
  }
}

// Cwise product and accumulate of two vectors. Since it's a MAC operation, the
// assumption here is that result array is initialized to valid values.
template <typename T>
inline void VectorVectorCwiseProductAccumulate(const T *__restrict__ vector1,
                                               const T *__restrict__ vector2, int v_size,
                                               T *__restrict__ result)
{
  for (int v = 0; v < v_size; v++)
  {
    *result++ += *vector1++ * *vector2++;
  }
}

// Cwise product of a vector and a batch-vector.
template <typename T>
inline void VectorBatchVectorCwiseProduct(const T *vector, int v_size, const T *batch_vector,
                                          int n_batch, T *result)
{
  for (int b = 0; b < n_batch; b++)
  {
    VectorVectorCwiseProduct(vector, batch_vector, v_size, result);
    // Update the pointers.
    result += v_size;
    batch_vector += v_size;
  }
}

// Cwise product and accumulate of a vector and a batch-vector. Since it's a MAC
// operation, the assumption here is that result array is initialized to valid
// values.
template <typename T>
inline void VectorBatchVectorCwiseProductAccumulate(const T *vector, int v_size,
                                                    const T *batch_vector, int n_batch, T *result)
{
  for (int b = 0; b < n_batch; b++)
  {
    VectorVectorCwiseProductAccumulate(vector, batch_vector, v_size, result);
    // Update the pointers.
    result += v_size;
    batch_vector += v_size;
  }
}

inline bool IsZeroVector(const float *vector, int v_size)
{
  return NEON_OR_PORTABLE(IsZeroVector, vector, v_size);
}

inline void ApplyActivationToVector(const float *vector, int v_size,
                                    FusedActivationFunctionType activation, float *result)
{
  PortableApplyActivationToVector(vector, v_size, activation, result);
}

inline void Sub1Vector(const float *vector, int v_size, float *result)
{
  NEON_OR_PORTABLE(Sub1Vector, vector, v_size, result);
}

inline void SymmetricQuantizeFloats(const float *values, const int size, int8_t *quantized_values,
                                    float *min, float *max, float *scaling_factor)
{
  return NEON_OR_PORTABLE(SymmetricQuantizeFloats, values, size, quantized_values, min, max,
                          scaling_factor);
}

inline void MatrixBatchVectorMultiplyAccumulate(const int8_t *matrix, const int m_rows,
                                                const int m_cols, const int8_t *vector,
                                                const float *scaling_factors, int n_batch,
                                                float *result, int result_stride)
{
  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols, vector,
                   scaling_factors, n_batch, result, result_stride);
}

inline void MatrixBatchVectorMultiplyAccumulate(const float *matrix, int m_rows, int m_cols,
                                                const float *vector, int n_batch, float *result,
                                                int result_stride)
{
  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols, vector, n_batch,
                   result, result_stride);
}

inline void MatrixBatchVectorMultiplyAccumulate(const int8_t *matrix, const int m_rows,
                                                const int m_cols, const int8_t *vectors,
                                                const float *scaling_factors, int n_batch,
                                                int32_t *scratch, float *result, int result_stride,
                                                ruy::Context *ruy_context)
{
  NEON_OR_PORTABLE(MatrixBatchVectorMultiplyAccumulate, matrix, m_rows, m_cols, vectors,
                   scaling_factors, n_batch, scratch, result, result_stride, ruy_context);
}

inline void MeanStddevNormalization(const float *input_vector, float *output_vector, int v_size,
                                    int n_batch)
{
  PortableMeanStddevNormalization(input_vector, output_vector, v_size, n_batch);
}

inline void ZeroVector(float *vector, int v_size) { PortableZeroVector(vector, v_size); }

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TENSOR_UTILS_H__
