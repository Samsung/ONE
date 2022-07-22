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
/*
The MIT License (MIT)

Copyright (c) 2017 Facebook Inc.
Copyright (c) 2017 Georgia Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "souschef/Data/Gaussian.h"
#include "souschef/LexicalCast.h"

#include <random>
#include <chrono>

#include <cassert>
#include <stdexcept>

namespace
{

// clang-format off
/*
This code block is from
- https://github.com/Maratyszcza/FP16/blob/0a92994d729ff76a58f692d3028ca1b64b145d91/include/fp16/fp16.h
- https://github.com/Maratyszcza/FP16/blob/0a92994d729ff76a58f692d3028ca1b64b145d91/include/fp16/bitcasts.h

TODO download from github source and remove this block
*/

static inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
  return as_float(w);
#elif defined(__CUDA_ARCH__)
  return __uint_as_float((unsigned int) w);
#elif defined(__INTEL_COMPILER)
  return _castu32_f32(w);
#else
  union {
    uint32_t as_bits;
    float as_value;
  } fp32 = { w };
  return fp32.as_value;
#endif
}

static inline uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
  return as_uint(f);
#elif defined(__CUDA_ARCH__)
  return (uint32_t) __float_as_uint(f);
#elif defined(__INTEL_COMPILER)
  return _castf32_u32(f);
#else
  union {
    float as_value;
    uint32_t as_bits;
  } fp32 = { f };
  return fp32.as_bits;
#endif

}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a 16-bit floating-point number in
 * IEEE half-precision format, in bit representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding mode and no operations on denormals)
 * floating-point operations and bitcasts between integer and floating-point variables.
 */
static inline uint16_t fp16_ieee_from_fp32_value(float f) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
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
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}
// clang-format on

} // namespace

namespace souschef
{

template <typename T>
static std::vector<uint8_t> generate_gaussian(int32_t count, float mean, float stddev,
                                              std::minstd_rand::result_type seed)
{
  std::minstd_rand rand{static_cast<std::minstd_rand::result_type>(seed)};
  std::normal_distribution<float> dist{mean, stddev};

  std::vector<uint8_t> res;

  constexpr float max_cap = std::numeric_limits<T>::max();
  constexpr float min_cap = std::numeric_limits<T>::min();
  for (uint32_t n = 0; n < count; ++n)
  {
    float raw_value = dist(rand);
    const float capped_value = std::max(min_cap, std::min(max_cap, raw_value));
    auto const value = static_cast<T>(capped_value);
    auto const arr = reinterpret_cast<const uint8_t *>(&value);

    for (uint32_t b = 0; b < sizeof(T); ++b)
    {
      res.emplace_back(arr[b]);
    }
  }

  return res;
}

template <typename T>
static std::vector<uint8_t> generate_gaussian(int32_t count, float mean, float stddev)
{
  auto time_stamp = std::chrono::system_clock::now().time_since_epoch().count();

  // Note this is implementation defined, change if needed.
  auto seed = static_cast<std::minstd_rand::result_type>(time_stamp);

  return generate_gaussian<T>(count, mean, stddev, seed);
}

std::vector<uint8_t> GaussianFloat32DataChef::generate(int32_t count) const
{
  return generate_gaussian<float>(count, _mean, _stddev);
}

std::vector<uint8_t> GaussianFloat16DataChef::generate(int32_t count) const
{
  auto time_stamp = std::chrono::system_clock::now().time_since_epoch().count();
  auto seed = static_cast<std::minstd_rand::result_type>(time_stamp);

  std::minstd_rand rand{static_cast<std::minstd_rand::result_type>(seed)};
  std::normal_distribution<float> dist{_mean, _stddev};

  std::vector<uint8_t> res;

  constexpr float max_cap = 1e9;
  constexpr float min_cap = -1e9;
  for (uint32_t n = 0; n < count; ++n)
  {
    float raw_value = dist(rand);
    const float capped_value = std::max(min_cap, std::min(max_cap, raw_value));
    const int16_t value = fp16_ieee_from_fp32_value(capped_value);
    auto const arr = reinterpret_cast<const uint8_t *>(&value);

    for (uint32_t b = 0; b < sizeof(int16_t); ++b)
    {
      res.emplace_back(arr[b]);
    }
  }

  return res;
}

std::vector<uint8_t> GaussianInt32DataChef::generate(int32_t count) const
{
  return generate_gaussian<int32_t>(count, _mean, _stddev);
}

std::vector<uint8_t> GaussianInt16DataChef::generate(int32_t count) const
{
  return generate_gaussian<int16_t>(count, _mean, _stddev);
}

std::vector<uint8_t> GaussianUint8DataChef::generate(int32_t count) const
{
  return generate_gaussian<uint8_t>(count, _mean, _stddev);
}

std::unique_ptr<DataChef> GaussianFloat32DataChefFactory::create(const Arguments &args) const
{
  if (args.count() != 2)
  {
    throw std::runtime_error{"invalid argument count: two arguments (mean/stddev) are expected"};
  }

  auto const mean = to_number<float>(args.value(0));
  auto const stddev = to_number<float>(args.value(1));

  return std::unique_ptr<DataChef>{new GaussianFloat32DataChef{mean, stddev}};
}

std::unique_ptr<DataChef> GaussianInt32DataChefFactory::create(const Arguments &args) const
{
  if (args.count() != 2)
  {
    throw std::runtime_error{"invalid argument count: two arguments (mean/stddev) are expected"};
  }

  auto const mean = to_number<float>(args.value(0));
  auto const stddev = to_number<float>(args.value(1));

  return std::unique_ptr<DataChef>{new GaussianInt32DataChef{mean, stddev}};
}

std::unique_ptr<DataChef> GaussianInt16DataChefFactory::create(const Arguments &args) const
{
  if (args.count() != 2)
  {
    throw std::runtime_error{"invalid argument count: two arguments (mean/stddev) are expected"};
  }

  auto const mean = to_number<float>(args.value(0));
  auto const stddev = to_number<float>(args.value(1));

  return std::unique_ptr<DataChef>{new GaussianInt16DataChef{mean, stddev}};
}

std::unique_ptr<DataChef> GaussianUint8DataChefFactory::create(const Arguments &args) const
{
  if (args.count() != 2)
  {
    throw std::runtime_error{"invalid argument count: two arguments (mean/stddev) are expected"};
  }

  auto const mean = to_number<float>(args.value(0));
  auto const stddev = to_number<float>(args.value(1));

  return std::unique_ptr<DataChef>{new GaussianUint8DataChef{mean, stddev}};
}

/**
 * @note As there is no float16, we use int16type for just random values
 * @todo Use FP16 library for real value
 */
std::unique_ptr<DataChef> GaussianFloat16DataChefFactory::create(const Arguments &args) const
{
  if (args.count() != 2)
  {
    throw std::runtime_error{"invalid argument count: two arguments (mean/stddev) are expected"};
  }

  auto const mean = to_number<float>(args.value(0));
  auto const stddev = to_number<float>(args.value(1));

  return std::unique_ptr<DataChef>{new GaussianFloat16DataChef{mean, stddev}};
}

} // namespace souschef
