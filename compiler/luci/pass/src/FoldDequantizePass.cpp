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

#include "luci/Pass/FoldDequantizePass.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

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
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit representation, to
 * a 32-bit floating-point number in IEEE single-precision format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding mode and no operations on denormals)
 * floating-point operations and bitcasts between integer and floating-point variables.
 */
static inline float fp16_ieee_to_fp32_value(uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0 - zero bits.
   */
  const uint32_t w = (uint32_t) h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the high bits of the 32-bit word:
   *
   *      +-----+------------+---------------------+
   *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
   *      +-----+------------+---------------------+
   * Bits  27-31    17-26            0-16
   */
  const uint32_t two_w = w + w;

  /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become mantissa and exponent
   * of a single-precision floating-point number:
   *
   *       S|Exponent |          Mantissa
   *      +-+---+-----+------------+----------------+
   *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
   *      +-+---+-----+------------+----------------+
   * Bits   | 23-31   |           0-22
   *
   * Next, there are some adjustments to the exponent:
   * - The exponent needs to be corrected by the difference in exponent bias between single-precision and half-precision
   *   formats (0x7F - 0xF = 0x70)
   * - Inf and NaN values in the inputs should become Inf and NaN values after conversion to the single-precision number.
   *   Therefore, if the biased exponent of the half-precision input was 0x1F (max possible value), the biased exponent
   *   of the single-precision output must be 0xFF (max possible value). We do this correction in two steps:
   *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset below) rather than by 0x70 suggested
   *     by the difference in the exponent bias (see above).
   *   - Then we multiply the single-precision result of exponent adjustment by 2**(-112) to reverse the effect of
   *     exponent adjustment by 0xE0 less the necessary exponent adjustment by 0x70 due to difference in exponent bias.
   *     The floating-point multiplication hardware would ensure than Inf and NaN would retain their value on at least
   *     partially IEEE754-compliant implementations.
   *
   * Note that the above operations do not handle denormal inputs (where biased exponent == 0). However, they also do not
   * operate on denormal inputs, and do not produce denormal results.
   */
  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  /*
   * Convert denormalized half-precision inputs into single-precision results (always normalized).
   * Zero inputs are also handled here.
   *
   * In a denormalized number the biased exponent is zero, and mantissa has on-zero bits.
   * First, we shift mantissa into bits 0-9 of the 32-bit word.
   *
   *                  zeros           |  mantissa
   *      +---------------------------+------------+
   *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
   *      +---------------------------+------------+
   * Bits             10-31                0-9
   *
   * Now, remember that denormalized half-precision numbers are represented as:
   *    FP16 = mantissa * 2**(-24).
   * The trick is to construct a normalized single-precision number with the same mantissa and thehalf-precision input
   * and with an exponent which would scale the corresponding mantissa bits to 2**(-24).
   * A normalized single-precision floating-point number is represented as:
   *    FP32 = (1 + mantissa * 2**(-23)) * 2**(exponent - 127)
   * Therefore, when the biased exponent is 126, a unit change in the mantissa of the input denormalized half-precision
   * number causes a change of the constructud single-precision number by 2**(-24), i.e. the same ammount.
   *
   * The last step is to adjust the bias of the constructed single-precision number. When the input half-precision number
   * is zero, the constructed single-precision number has the value of
   *    FP32 = 1 * 2**(126 - 127) = 2**(-1) = 0.5
   * Therefore, we need to subtract 0.5 from the constructed single-precision number to get the numerical equivalent of
   * the input half-precision number.
   */
  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  /*
   * - Choose either results of conversion of input as a normalized number, or as a denormalized number, depending on the
   *   input exponent. The variable two_w contains input exponent in bits 27-31, therefore if its smaller than 2**27, the
   *   input is either a denormal number, or zero.
   * - Combine the result of conversion of exponent and mantissa with the sign of the input number.
   */
  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result = sign |
    (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}
// clang-format on

} // namespace

namespace
{

bool is_hybrid_kernel_supported(loco::Node *node)
{
  if (dynamic_cast<luci::CircleFullyConnected *>(node) != nullptr)
    return true;

  return false;
}

bool is_foldable_const(luci::CircleConst *node)
{
  if (node->dtype() == loco::DataType::FLOAT16)
    return true;

  if (node->quantparam() == nullptr)
    return false;

  if (node->dtype() == loco::DataType::S8)
    return true;
  if (node->dtype() == loco::DataType::U8)
    return true;
  if (node->dtype() == loco::DataType::S16)
    return true;
  if (node->dtype() == loco::DataType::S32)
    return true;
  if (node->dtype() == loco::DataType::S64)
    return true;

  return false;
}

luci::CircleConst *dequantized_const_node(luci::CircleConst *const_node)
{
  auto name = const_node->name();
  assert(name.length() > 0);
  auto g = const_node->graph();
  auto new_const_node = g->nodes()->create<luci::CircleConst>();

  new_const_node->dtype(loco::DataType::FLOAT32);
  new_const_node->rank(const_node->rank());
  uint32_t dim_size = 1;
  for (uint32_t i = 0; i < new_const_node->rank(); ++i)
  {
    new_const_node->dim(i) = const_node->dim(i);
    dim_size *= const_node->dim(i).value();
  }
  new_const_node->size<loco::DataType::FLOAT32>(dim_size);
  new_const_node->shape_status(luci::ShapeStatus::VALID);
  new_const_node->name(name + "_DQ");

  if (const_node->dtype() == loco::DataType::FLOAT16)
  {
    for (uint32_t i = 0; i < new_const_node->size<loco::DataType::FLOAT32>(); ++i)
    {
      auto raw = const_node->at<loco::DataType::FLOAT16>(i);
      new_const_node->at<loco::DataType::FLOAT32>(i) = fp16_ieee_to_fp32_value(raw);
    }
    return new_const_node;
  }

  if (const_node->quantparam() == nullptr)
  {
    throw std::runtime_error("Given constant node has no quantization parameter");
  }

  const int32_t q_dim = const_node->quantparam()->quantized_dimension;
  // For scalar, q_dim_value is 1
  // For non-scalar, q_dim_value is the size of quantized dimension
  const int32_t q_dim_value = const_node->rank() == 0 ? 1 : const_node->dim(q_dim).value();

  int32_t right_count = q_dim_value;
  for (uint32_t i = q_dim + 1; i < const_node->rank(); ++i)
    right_count *= const_node->dim(i).value();

  for (uint32_t i = 0; i < new_const_node->size<loco::DataType::FLOAT32>(); ++i)
  {
    uint32_t qd = (i % right_count) / (right_count / q_dim_value);
    if (qd >= const_node->quantparam()->zerop.size())
      qd = 0;

    switch (const_node->dtype())
    {
      case loco::DataType::S8:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::S8>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      case loco::DataType::S16:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::S16>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      case loco::DataType::S32:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::S32>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      case loco::DataType::S64:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::S64>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      case loco::DataType::U8:
        new_const_node->at<loco::DataType::FLOAT32>(i) =
          static_cast<float>(const_node->at<loco::DataType::U8>(i) -
                             const_node->quantparam()->zerop.at(qd)) *
          const_node->quantparam()->scale.at(qd);
        break;
      default:
        throw std::runtime_error("Not supported dtype for FoldDequantizePass");
    }
  }

  return new_const_node;
}

bool replace_const_node(loco::Node *node, luci::CircleConst *const_node)
{
  if (auto gather = dynamic_cast<luci::CircleGather *>(node))
  {
    gather->params(dequantized_const_node(const_node));
    gather->dtype(loco::DataType::FLOAT32);
    return true;
  }
  else
  {
    // TODO Support more ops
    return false;
  }
}

} // namespace

namespace luci
{

/**
 *
 * Folding pattern 1 - When input of Dequantize is foldable constant
 *
 * [Before]
 *     quantized_const_input ---------- Dequantize ---------- Op ---
 *                             +-- Op1_with_quant_input ---
 *                             +-- Op2_with_quant_input ---
 *
 * [After]
 *   dequantized_const_input -------------------------------- Op ---
 *
 *     quantized_const_input ----- Op1_with_quant_input ---
 *                             +-- Op2_with_quant_input ---
 *
 *
 * Folding pattern 2 - When input of Dequantize uses quantized output value
 *
 * [Before]
 *     quantized_const_input ----- Gather ----- Dequantize --- Op ---
 *                             +-- Op1_with_quant_input ---
 *                             +-- Op2_with_quant_input ---
 *
 * [After]
 *   dequantized_const_input ------Gather -------------------- Op ---
 *
 *     quantized_const_input ----- Op1_with_quant_input ---
 *                             +-- Op2_with_quant_input ---
 *
 *
 */
bool FoldDequantizePass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto circle_dequant = dynamic_cast<luci::CircleDequantize *>(node))
    {
      if (auto const_input = dynamic_cast<luci::CircleConst *>(circle_dequant->input()))
      {
        // Pattern 1 - When input of Dequantize is foldable constant
        if (is_foldable_const(const_input))
        {
          loco::replace(circle_dequant).with(dequantized_const_node(const_input));
          changed = true;
        }
      }
    }
    else if (auto const_node = dynamic_cast<luci::CircleConst *>(node))
    {
      if (is_foldable_const(const_node))
      {
        for (auto const_node_user : loco::succs(const_node))
        {
          // If user is hybrid kernel supported operation, do not dequantize
          if (is_hybrid_kernel_supported(const_node_user))
            continue;

          auto users = loco::succs(const_node_user);
          if (users.size() > 1)
            continue;

          // Pattern 2 - When input of Dequantize uses quantized output value
          if (auto dequant = dynamic_cast<luci::CircleDequantize *>(*users.begin()))
          {
            if (replace_const_node(const_node_user, const_node))
            {
              loco::replace(dequant).with(const_node_user);
              luci::add_origin(loco::must_cast<luci::CircleNode *>(const_node_user),
                               luci::get_origin(dequant));
              changed = true;
            }
          }
        }
      }
    }
  }

  return changed;
}

} // namespace luci
