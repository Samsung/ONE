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

#ifndef __NNFW_CKER_PORTABLE_TENSOR_UTILS_H__
#define __NNFW_CKER_PORTABLE_TENSOR_UTILS_H__

#include "cker/Types.h"
#include "cker/neon/neon_check.h"
#include <ruy/context.h>

#include <cstring>
#include <cmath>

namespace nnfw
{
namespace cker
{

class ActivationFunctor
{
public:
  explicit ActivationFunctor(FusedActivationFunctionType act) : act_(act) {}

  float operator()(float a) const
  {
    switch (act_)
    {
      case FusedActivationFunctionType::kNone:
        return a;
      case FusedActivationFunctionType::kRelu:
        return a < 0.f ? 0.f : a;
      case FusedActivationFunctionType::kRelu6:
        return std::max(0.f, std::min(a, 6.f));
      case FusedActivationFunctionType::kTanh:
        return std::tanh(a);
      case FusedActivationFunctionType::kSigmoid:
        return 1.0f / (1.0f + std::exp(-a));
      default:
        // TODO(aselle): More informative fatal error!
        exit(1);
    }
  }

private:
  FusedActivationFunctionType act_;
};

template <typename T>
void PortableCwiseClipping(T *vector, const int v_size, const T clipping_value)
{
  for (int i = 0; i < v_size; i++)
  {
    vector[i] = std::max(std::min(clipping_value, vector[i]), static_cast<T>(-clipping_value));
  }
}

inline void PortableVectorBatchVectorAssign(const float *vector, int v_size, int n_batch,
                                            float *batch_vector)
{
  for (int b = 0; b < n_batch; b++)
  {
    memcpy(batch_vector + b * v_size, vector, v_size * sizeof(float));
  }
}

inline void PortableVectorBatchVectorAdd(const float *vector, int v_size, int n_batch,
                                         float *batch_vector)
{
  for (int b = 0; b < n_batch; b++)
  {
    for (int i = 0; i < v_size; ++i)
    {
      batch_vector[i] += vector[i];
    }
    batch_vector += v_size;
  }
}

inline bool PortableIsZeroVector(const float *vector, int v_size)
{
  for (int i = 0; i < v_size; ++i)
  {
    if (*vector++ != 0.0f)
      return false;
  }
  return true;
}

inline void PortableApplyActivationToVector(const float *vector, int v_size,
                                            FusedActivationFunctionType activation, float *result)
{
  auto activation_func = ActivationFunctor(activation);
  for (int v = 0; v < v_size; v++)
  {
    *result++ = (activation_func)(*vector++);
  }
}

inline void PortableSub1Vector(const float *vector, int v_size, float *result)
{
  for (int v = 0; v < v_size; v++)
  {
    *result++ = 1.0f - *vector++;
  }
}

inline void PortableSymmetricQuantizeFloats(const float *values, const int size,
                                            int8_t *quantized_values, float *min_value,
                                            float *max_value, float *scaling_factor)
{
  auto minmax = std::minmax_element(values, values + size);
  *min_value = *minmax.first;
  *max_value = *minmax.second;
  const int kScale = 127;
  const float range = std::max(std::abs(*min_value), std::abs(*max_value));
  if (range == 0)
  {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;
  for (int i = 0; i < size; ++i)
  {
    const int32_t quantized_value =
      static_cast<int32_t>(std::round(values[i] * scaling_factor_inv));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = std::min(kScale, std::max(-kScale, quantized_value));
  }
}

inline void PortableAsymmetricQuantizeFloats(const float *values, const int size,
                                             int8_t *quantized_values, float *scaling_factor,
                                             int32_t *offset)
{
  /* Copied from TensorFlow PortableAsymmetricQuantizeFloats */
  const int32_t kMinScale = -128;
  const int32_t kMaxScale = 127;
  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  const auto minmax = std::minmax_element(values, values + size);
  const double rmin = static_cast<double>(std::min(0.0f, *minmax.first));
  const double rmax = static_cast<double>(std::max(0.0f, *minmax.second));
  if (rmin == rmax)
  {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    *offset = 0;
    return;
  }
  else
  {
    double scale = (rmax - rmin) / (qmax_double - qmin_double);
    const double zero_point_from_min = qmin_double - rmin / scale;
    const double zero_point_from_max = qmax_double - rmax / scale;
    const double zero_point_from_min_error = std::abs(qmin_double) + std::abs(rmin / scale);
    const double zero_point_from_max_error = std::abs(qmax_double) + std::abs(rmax / scale);
    const double zero_point_double = zero_point_from_min_error < zero_point_from_max_error
                                       ? zero_point_from_min
                                       : zero_point_from_max;
    int8_t nudged_zero_point = 0;
    if (zero_point_double <= qmin_double)
    {
      nudged_zero_point = kMinScale;
    }
    else if (zero_point_double >= qmax_double)
    {
      nudged_zero_point = kMaxScale;
    }
    else
    {
      nudged_zero_point = static_cast<int8_t>(round(zero_point_double));
    }
    *scaling_factor = scale;
    *offset = nudged_zero_point;
  }
  const float scaling_factor_inv = 1.0f / *scaling_factor;
  for (int i = 0; i < size; ++i)
  {
    const int32_t quantized_value =
      static_cast<int32_t>(std::round(*offset + values[i] * scaling_factor_inv));
    quantized_values[i] = std::min(kMaxScale, std::max(kMinScale, quantized_value));
  }
}

inline void PortableMatrixBatchVectorMultiplyAccumulate(const int8_t *__restrict__ matrix,
                                                        const int m_rows, const int m_cols,
                                                        const int8_t *__restrict__ vectors,
                                                        const float *scaling_factors, int n_batch,
                                                        float *__restrict__ result,
                                                        int result_stride)
{
  int batch, row, col;
  for (batch = 0; batch < n_batch; ++batch, vectors += m_cols)
  {
    const float batch_scaling_factor = scaling_factors[batch];
    // Get the address of the first row.
    const int8_t *row_ptr = matrix;
    for (row = 0; row < m_rows; ++row, result += result_stride)
    {
      // Initialize the dot product sum for the row to 0.
      int32_t dotprod = 0;
#if defined(__GNUC__)
      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */, 3 /* temporal locality */);
#endif
      for (col = 0; col < m_cols; ++col, ++row_ptr)
      {
        dotprod += (*row_ptr) * (vectors[col]);
      } // for col
      *result += (dotprod * batch_scaling_factor);
    } // for row
  }   // for batch
}

inline void PortableMatrixBatchVectorMultiplyAccumulate(const int8_t *__restrict__ matrix,
                                                        const int m_rows, const int m_cols,
                                                        const int8_t *__restrict__ vector,
                                                        const float *scaling_factors, int n_batch,
                                                        int32_t *, float *__restrict__ result,
                                                        int result_stride, ruy::Context *)
{
  PortableMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vector, scaling_factors,
                                              n_batch, result, result_stride);
}

inline void PortableMatrixBatchVectorMultiplyAccumulate(const float *matrix, int m_rows, int m_cols,
                                                        const float *vector, int n_batch,
                                                        float *result, int result_stride)
{
  float *result_in_batch = result;
  for (int b = 0; b < n_batch; b++)
  {
    const float *matrix_ptr = matrix;
    for (int r = 0; r < m_rows; r++)
    {
      float dot_prod = 0.0f;
      const float *vector_in_batch = vector + b * m_cols;
      for (int c = 0; c < m_cols; c++)
      {
        dot_prod += *matrix_ptr++ * *vector_in_batch++;
      }
      *result_in_batch += dot_prod;
      result_in_batch += result_stride;
    }
  }
}

inline void PortableMeanStddevNormalization(const float *input_vector, float *output_vector,
                                            int v_size, int n_batch)
{
  for (int batch = 0; batch < n_batch; ++batch)
  {
    float sum = 0.0f;
    for (int i = 0; i < v_size; ++i)
    {
      sum += input_vector[i];
    }
    const float mean = sum / v_size;
    float sum_diff_sq = 0.0f;
    for (int i = 0; i < v_size; ++i)
    {
      const float diff = input_vector[i] - mean;
      sum_diff_sq += diff * diff;
    }
    const float variance = sum_diff_sq / v_size;
    constexpr float kNormalizationConstant = 1e-8f;
    const float stddev_inv = 1.0f / std::sqrt(variance + kNormalizationConstant);
    for (int i = 0; i < v_size; ++i)
    {
      output_vector[i] = (input_vector[i] - mean) * stddev_inv;
    }
    input_vector += v_size;
    output_vector += v_size;
  }
}

inline void PortableZeroVector(float *vector, int v_size) { std::fill_n(vector, v_size, 0); }

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_PORTABLE_TENSOR_UTILS_H__
