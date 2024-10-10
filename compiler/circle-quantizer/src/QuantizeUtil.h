/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright (c) 2023 Georgi Gerganov
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

#ifndef LUCI_QUANTIZE_QUANTIZE_UTIL_H
#define LUCI_QUANTIZE_QUANTIZE_UTIL_H

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cmath>

// Copy from llama.cpp

typedef uint16_t ggml_fp16_t;

#define QK4_0 32
typedef struct
{
  ggml_fp16_t d;         // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#define QK8_0 32
typedef struct
{
  ggml_fp16_t d;    // delta
  int8_t qs[QK8_0]; // quants
} block_q8_0;

union block_q4_0_u {
  uint8_t u8[sizeof(block_q4_0)];
  block_q4_0 b;
};

union block_q8_0_u {
  uint8_t u8[sizeof(block_q8_0)];
  block_q8_0 b;
};

static inline uint32_t fp32_to_bits(float f)
{
  union {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

static inline float fp32_from_bits(uint32_t w)
{
  union {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f)
{
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;

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

#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

void quantize_row_q4_0_reference(const float *x, block_q4_0 *y, int k)
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

    y[i].d = GGML_FP32_TO_FP16(d);

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

size_t ggml_quantize_q4_0(const float *src, void *dst, int n, int k)
{
  assert(k % QK4_0 == 0);
  const int nb = k / QK4_0;

  for (int b = 0; b < n; b += k)
  {
    block_q4_0 *y = (block_q4_0 *)dst + b / QK4_0;

    quantize_row_q4_0_reference(src + b, y, k);

    for (int i = 0; i < nb; i++)
    {
      for (int j = 0; j < QK4_0; j += 2)
      {
        const uint8_t vi0 = y[i].qs[j / 2] & 0x0F;
        const uint8_t vi1 = y[i].qs[j / 2] >> 4;
      }
    }
  }

  return (n / QK4_0 * sizeof(block_q4_0));
}

void quantize_row_q8_0_reference(const float *x, block_q8_0 *y, int k)
{
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  for (int i = 0; i < nb; i++)
  {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++)
    {
      const float v = x[i * QK8_0 + j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; ++j)
    {
      const float x0 = x[i * QK8_0 + j] * id;

      y[i].qs[j] = roundf(x0);
    }
  }
}

size_t ggml_quantize_q8_0(const float *src, void *dst, int n, int k)
{
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  for (int b = 0; b < n; b += k)
  {
    block_q8_0 *y = (block_q8_0 *)dst + b / QK8_0;

    quantize_row_q8_0_reference(src + b, y, k);

    for (int i = 0; i < nb; i++)
    {
      for (int j = 0; j < QK8_0; ++j)
      {
        const int8_t vi = y[i].qs[j];
      }
    }
  }

  return (n / QK8_0 * sizeof(block_q8_0));
}

#endif // LUCI_QUANTIZE_QUANTIZE_UTIL_H
