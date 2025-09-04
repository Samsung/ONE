
/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "common.h"

#include <cmath>

//
// Quantization code from ggml
//
namespace
{

#define QK4_0 32

typedef uint16_t ggml_fp16_t;

typedef struct
{
  ggml_fp16_t d;         // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#if defined(__ARM_NEON)

#include <arm_neon.h>

#ifdef _MSC_VER

typedef uint16_t ggml_fp16_internal_t;

#else

typedef __fp16 ggml_fp16_internal_t;

#endif // _MSC_VER
#endif // defined(__ARM_NEON)

#if defined(__ARM_NEON) && !defined(_MSC_VER)

inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f)
{
  ggml_fp16_t res;
  ggml_fp16_internal_t tmp = f;
  memcpy(&res, &tmp, sizeof(ggml_fp16_t));
  return res;
}

#else

inline float fp32_from_bits(uint32_t w)
{
  union {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

inline uint32_t fp32_to_bits(float f)
{
  union {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f)
{
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || \
  defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
#else
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000))
  {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void quantize_row_q4_0_ref(const float *x, block_q4_0 *y, int64_t k)
{
  static const int qk = QK4_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++)
  {
    float amax = 0.0f; // absolute max
    float max = 0.0f;

    for (int j = 0; j < qk; j++)
    {
      const float v = x[i * qk + j];
      if (amax < fabsf(v))
      {
        amax = fabsf(v);
        max = v;
      }
    }

    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = ggml_compute_fp32_to_fp16(d);

    for (int j = 0; j < qk / 2; ++j)
    {
      const float x0 = x[i * qk + 0 + j] * id;
      const float x1 = x[i * qk + qk / 2 + j] * id;

      const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
      const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));
      y[i].qs[j] = xi0;
      y[i].qs[j] |= xi1 << 4;
    }
  }
}

void quantize_q4_0(const float *src, void *dst, int64_t nrow, int64_t n_per_row)
{
  quantize_row_q4_0_ref(src, reinterpret_cast<block_q4_0 *>(dst), (int64_t)nrow * n_per_row);
}

} // namespace

bool tensorInfoEqual(const nnfw_tensorinfo &info1, const nnfw_tensorinfo &info2)
{
  if (info1.dtype != info2.dtype)
    return false;
  if (info1.rank != info2.rank)
    return false;
  for (int i = 0; i < info1.rank; i++)
    if (info1.dims[i] != info2.dims[i])
      return false;
  return true;
}

uint64_t tensorInfoNumElements(const nnfw_tensorinfo &ti)
{
  uint64_t n = 1;
  for (int32_t i = 0; i < ti.rank; ++i)
  {
    n *= static_cast<uint64_t>(ti.dims[i]);
  }
  return n;
}

std::vector<uint8_t> quantData(const std::vector<float> &buf_val, const circle::TensorType type)
{
  switch (type)
  {
    case circle::TensorType::TensorType_GGML_Q4_0:
    {
      size_t num_elems = buf_val.size();
      const size_t block_size = QK4_0;
      const int64_t num_block = num_elems / block_size;
      const size_t block_struct_size = sizeof(block_q4_0);

      auto buf = std::vector<uint8_t>(num_block * block_struct_size);
      quantize_q4_0(buf_val.data(), buf.data(), 1, num_elems);

      return buf;
    }
    default:
      throw std::runtime_error("Unsupported tensor type");
  }
}
