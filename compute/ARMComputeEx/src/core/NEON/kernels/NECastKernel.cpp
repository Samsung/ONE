/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

/*
 * Copyright (c) 2017-2019 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "arm_compute/core/NEON/kernels/NECastKernel.h"

#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/NEON/NEAsymm.h"
#include "arm_compute/core/NEON/wrapper/wrapper.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output,
                          SubDataType input_subtype)
{
  ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(input, 1, DataType::U8, DataType::S8,
                                                       DataType::QASYMM8, DataType::U32,
                                                       DataType::S32, DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON(input_subtype == SubDataType::BOOL &&
                              input->data_type() != DataType::U8);

  if (output->tensor_shape().total_size() > 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(output);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output, 1, DataType::U8, DataType::S8,
                                                         DataType::QASYMM8, DataType::U32,
                                                         DataType::S32, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
  }

  return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
  // Configure kernel window
  Window win = calculate_max_window(*input, Steps());

  // Output tensor auto initialization if not yet initialized
  auto_init_if_empty(*output, input->tensor_shape(), 1, DataType::F32);

  // NECastKernel doesn't need padding so update_window_and_padding() can be skipped
  Coordinates coord;
  coord.set_num_dimensions(output->num_dimensions());
  output->set_valid_region(ValidRegion(coord, output->tensor_shape()));

  return std::make_tuple(Status{}, win);
}

typedef struct bool8x16
{
  uint8x16_t val;
} bool8x16_t;

static inline uint8x16_t vreinterpretq_u8_b8(bool8x16_t __a) { return (uint8x16_t)__a.val; }

template <typename ToV, typename FromV> inline ToV vcast(const FromV &v) { return v; }
template <> inline uint8x16_t vcast(const bool8x16_t &v)
{
  const uint8x16_t vu8 = vreinterpretq_u8_b8(v);
  const uint8x16_t zero_uint8x16 = vdupq_n_u8(0);
  uint8x16_t mask = vcgtq_u8(vu8, zero_uint8x16);
  return vshrq_n_u8(mask, 7); // true -> 1, false -> 0
}

template <> inline uint32x4x4_t vcast(const bool8x16_t &v)
{
  const uint8x16_t vu8 = vreinterpretq_u8_b8(v);
  const uint8x16_t zero_uint8x16 = vdupq_n_u8(0);
  uint8x16_t mask = vcgtq_u8(vu8, zero_uint8x16);
  uint8x16_t vb = vshrq_n_u8(mask, 7); // true -> 1, false -> 0

  const uint32x4x4_t ret = {{
      vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(vb)))),
      vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(vb)))),
      vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(vb)))),
      vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(vb)))),
  }};

  return ret;
}

template <> inline int32x4x4_t vcast(const bool8x16_t &v)
{
  const uint8x16_t vu8 = vreinterpretq_u8_b8(v);
  const uint8x16_t zero_uint8x16 = vdupq_n_u8(0);
  uint8x16_t mask = vcgtq_u8(vu8, zero_uint8x16);
  uint8x16_t vb = vshrq_n_u8(mask, 7); // true -> 1, false -> 0

  const int32x4x4_t ret = {{
      vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(vb))))),
      vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(vb))))),
      vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(vb))))),
      vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(vb))))),
  }};

  return ret;
}

template <> inline float32x4x4_t vcast(const bool8x16_t &v)
{
  const uint8x16_t vu8 = vreinterpretq_u8_b8(v);
  const uint8x16_t zero_uint8x16 = vdupq_n_u8(0);
  uint8x16_t mask = vcgtq_u8(vu8, zero_uint8x16);
  uint8x16_t vb = vshrq_n_u8(mask, 7); // true -> 1, false -> 0

  const float32x4x4_t ret = {{
      vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(vb))))),
      vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(vb))))),
      vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(vb))))),
      vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(vb))))),
  }};

  return ret;
}

template <> inline uint32x4x4_t vcast(const uint8x16_t &v)
{
  const uint32x4x4_t ret = {{
      vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(v)))),
      vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(v)))),
      vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(v)))),
      vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(v)))),
  }};

  return ret;
}

template <> inline int32x4x4_t vcast(const uint8x16_t &v)
{
  const int32x4x4_t ret = {{
      vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(v))))),
      vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(v))))),
      vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(v))))),
      vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(v))))),
  }};

  return ret;
}

template <> inline float32x4x4_t vcast(const uint8x16_t &v)
{
  const float32x4x4_t ret = {{
      vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(v))))),
      vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_low_u8(v))))),
      vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_high_u8(v))))),
      vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(vget_high_u8(v))))),
  }};

  return ret;
}

template <> inline uint8x16_t vcast(const int32x4x4_t &v)
{
  // Saturate cast
  return vcombine_u8(vqmovn_u16(vcombine_u16(vqmovun_s32(v.val[0]), vqmovun_s32(v.val[1]))),
                     vqmovn_u16(vcombine_u16(vqmovun_s32(v.val[2]), vqmovun_s32(v.val[3]))));
}

template <> inline uint32x4x4_t vcast(const int32x4x4_t &v)
{
  // Saturate cast
  const uint32x4x4_t ret = {{
      vcombine_u32(vqmovun_s64(vmovl_s32(vget_low_s32(v.val[0]))),
                   vqmovun_s64(vmovl_s32(vget_high_s32(v.val[0])))),
      vcombine_u32(vqmovun_s64(vmovl_s32(vget_low_s32(v.val[1]))),
                   vqmovun_s64(vmovl_s32(vget_high_s32(v.val[1])))),
      vcombine_u32(vqmovun_s64(vmovl_s32(vget_low_s32(v.val[2]))),
                   vqmovun_s64(vmovl_s32(vget_high_s32(v.val[2])))),
      vcombine_u32(vqmovun_s64(vmovl_s32(vget_low_s32(v.val[3]))),
                   vqmovun_s64(vmovl_s32(vget_high_s32(v.val[3])))),
  }};

  return ret;
}

template <> inline float32x4x4_t vcast(const int32x4x4_t &v)
{
  const float32x4x4_t ret = {{
      vcvtq_f32_s32(v.val[0]), vcvtq_f32_s32(v.val[1]), vcvtq_f32_s32(v.val[2]),
      vcvtq_f32_s32(v.val[3]),
  }};

  return ret;
}

template <> inline uint8x16_t vcast(const uint32x4x4_t &v)
{
  return vcombine_u8(vqmovn_u16(vcombine_u16(vqmovn_u32(v.val[0]), vqmovn_u32(v.val[1]))),
                     vqmovn_u16(vcombine_u16(vqmovn_u32(v.val[2]), vqmovn_u32(v.val[3]))));
}

template <> inline int32x4x4_t vcast(const uint32x4x4_t &v)
{
  const int32x4x4_t ret = {{
      vcombine_s32(vmovn_s64(vreinterpretq_s64_u64(vmovl_u32(vget_low_u32(v.val[0])))),
                   vmovn_s64(vreinterpretq_s64_u64(vmovl_u32(vget_high_u32(v.val[0]))))),
      vcombine_s32(vmovn_s64(vreinterpretq_s64_u64(vmovl_u32(vget_low_u32(v.val[1])))),
                   vmovn_s64(vreinterpretq_s64_u64(vmovl_u32(vget_high_u32(v.val[1]))))),
      vcombine_s32(vmovn_s64(vreinterpretq_s64_u64(vmovl_u32(vget_low_u32(v.val[2])))),
                   vmovn_s64(vreinterpretq_s64_u64(vmovl_u32(vget_high_u32(v.val[2]))))),
      vcombine_s32(vmovn_s64(vreinterpretq_s64_u64(vmovl_u32(vget_low_u32(v.val[3])))),
                   vmovn_s64(vreinterpretq_s64_u64(vmovl_u32(vget_high_u32(v.val[3]))))),
  }};

  return ret;
}

template <> inline float32x4x4_t vcast(const uint32x4x4_t &v)
{
  const float32x4x4_t ret = {{
      vcvtq_f32_u32(v.val[0]), vcvtq_f32_u32(v.val[1]), vcvtq_f32_u32(v.val[2]),
      vcvtq_f32_u32(v.val[3]),
  }};

  return ret;
}

template <> inline uint8x16_t vcast(const float32x4x4_t &v)
{
  // Saturate cast
  return vcombine_u8(vqmovn_u16(vcombine_u16(vqmovun_s32(vcvtq_s32_f32(v.val[0])),
                                             vqmovun_s32(vcvtq_s32_f32(v.val[1])))),
                     vqmovn_u16(vcombine_u16(vqmovun_s32(vcvtq_s32_f32(v.val[2])),
                                             vqmovun_s32(vcvtq_s32_f32(v.val[3])))));
}

template <> inline uint32x4x4_t vcast(const float32x4x4_t &v)
{
  const uint32x4x4_t ret = {{
      vcvtq_u32_f32(v.val[0]), vcvtq_u32_f32(v.val[1]), vcvtq_u32_f32(v.val[2]),
      vcvtq_u32_f32(v.val[3]),
  }};

  return ret;
}

template <> inline int32x4x4_t vcast(const float32x4x4_t &v)
{
  const int32x4x4_t ret = {{
      vcvtq_s32_f32(v.val[0]), vcvtq_s32_f32(v.val[1]), vcvtq_s32_f32(v.val[2]),
      vcvtq_s32_f32(v.val[3]),
  }};

  return ret;
}

template <typename T> struct cast_vector;
template <> struct cast_vector<bool>
{
  using type = bool8x16_t;
};
template <> struct cast_vector<uint8_t>
{
  using type = uint8x16_t;
};
template <> struct cast_vector<uint32_t>
{
  using type = uint32x4x4_t;
};
template <> struct cast_vector<int32_t>
{
  using type = int32x4x4_t;
};
template <> struct cast_vector<float>
{
  using type = float32x4x4_t;
};

template <typename T> inline void store_result(T *ptr, const typename cast_vector<T>::type &v)
{
  wrapper::vstore(ptr, v.val[0]);
  wrapper::vstore(ptr + 4, v.val[1]);
  wrapper::vstore(ptr + 8, v.val[2]);
  wrapper::vstore(ptr + 12, v.val[3]);
}

template <> inline void store_result<uint8_t>(uint8_t *ptr, const uint8x16_t &v)
{
  wrapper::vstore(ptr, v);
}

inline bool8x16_t vloadq(const bool *ptr)
{
  bool8x16_t ret;
  ret.val = wrapper::vloadq(reinterpret_cast<const uint8_t *>(ptr));
  return ret;
}

template <typename T> inline typename cast_vector<T>::type load_input(const T *ptr)
{
  return wrapper::vloadq(ptr);
}

template <> inline typename cast_vector<bool>::type load_input(const bool *ptr)
{
  return vloadq(ptr);
}

template <> inline typename cast_vector<uint32_t>::type load_input(const uint32_t *ptr)
{
  return vld4q_u32(ptr);
}

template <> inline typename cast_vector<int32_t>::type load_input(const int32_t *ptr)
{
  return vld4q_s32(ptr);
}

template <> inline typename cast_vector<float>::type load_input(const float *ptr)
{
  return vld4q_f32(ptr);
}

template <typename T> inline T get_value(const T *ptr) { return *ptr; }

template <> inline bool get_value(const bool *ptr)
{
  bool ret = (*ptr != 0);
  return ret;
}

template <typename FromT> void run_cast(const ITensor *input, ITensor *output, const Window &window)
{
  const int window_step_x = 16;
  const auto window_start_x = static_cast<int>(window.x().start());
  const auto window_end_x = static_cast<int>(window.x().end());

  // Collapse window and reset first dimension to handle tail calculations manually
  Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
  win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

  // Create iterators
  Iterator in(input, win_collapsed);
  Iterator out(output, win_collapsed);

#ifdef __aarch64__
  constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_EVEN;
#else  //__aarch64__
  constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_ZERO;
#endif //__aarch64__

  execute_window_loop(
      win_collapsed,
      [&](const Coordinates &) {
        const auto in_ptr = reinterpret_cast<const FromT *>(in.ptr());

        int x = window_start_x;
        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
          using from_vector = typename cast_vector<FromT>::type;
          const from_vector vin = load_input(in_ptr + x);

          switch (output->info()->data_type())
          {
            case DataType::U8:
            {
              using to_vector = typename cast_vector<uint8_t>::type;
              const to_vector vout = vcast<to_vector, from_vector>(vin);
              store_result<uint8_t>(reinterpret_cast<uint8_t *>(out.ptr()) + x, vout);
              break;
            }
            case DataType::QASYMM8:
            {
              using to_vector = typename cast_vector<float>::type;
              const UniformQuantizationInfo &qinfo_out =
                  output->info()->quantization_info().uniform();
              const auto vf = vcast<to_vector, from_vector>(vin);
              const auto vout = vquantize(vf, qinfo_out);
              store_result<qasymm8_t>(reinterpret_cast<qasymm8_t *>(out.ptr()) + x, vout);
              break;
            }
            case DataType::U32:
            {
              using to_vector = typename cast_vector<uint32_t>::type;
              const to_vector vout = vcast<to_vector, from_vector>(vin);
              store_result<uint32_t>(reinterpret_cast<uint32_t *>(out.ptr()) + x, vout);
              break;
            }
            case DataType::S32:
            {
              using to_vector = typename cast_vector<int32_t>::type;
              const to_vector vout = vcast<to_vector, from_vector>(vin);
              store_result<int32_t>(reinterpret_cast<int32_t *>(out.ptr()) + x, vout);
              break;
            }
            case DataType::F32:
            {
              using to_vector = typename cast_vector<float>::type;
              const to_vector vout = vcast<to_vector, from_vector>(vin);
              store_result<float>(reinterpret_cast<float *>(out.ptr()) + x, vout);
              break;
            }
            default:
              ARM_COMPUTE_ERROR("Unsupported data type.");
          }
        }

        // Compute left-over elements
        for (; x < window_end_x; ++x)
        {
          FromT val = get_value(in_ptr + x);
          switch (output->info()->data_type())
          {
            case DataType::U8:
            {
              *(reinterpret_cast<uint8_t *>(out.ptr()) + x) = static_cast<uint8_t>(val);
              break;
            }
            case DataType::QASYMM8:
            {
              const QuantizationInfo &qinfo_out = output->info()->quantization_info();
              const auto qval =
                  quantize_qasymm8(static_cast<float>(val), qinfo_out, rounding_policy);
              *(reinterpret_cast<qasymm8_t *>(out.ptr()) + x) = qval;
              break;
            }
            case DataType::U32:
            {
              *(reinterpret_cast<uint32_t *>(out.ptr()) + x) = static_cast<uint32_t>(val);
              break;
            }
            case DataType::S32:
            {
              *(reinterpret_cast<int32_t *>(out.ptr()) + x) = static_cast<int32_t>(val);
              break;
            }
            case DataType::F32:
            {
              *(reinterpret_cast<float *>(out.ptr()) + x) = static_cast<float>(val);
              break;
            }
            default:
              ARM_COMPUTE_ERROR("Unsupported data type.");
          }
        }
      },
      in, out);
}

void run_cast_qasymm8(const ITensor *input, ITensor *output, const Window &window)
{
  const int window_step_x = 16;
  const auto window_start_x = static_cast<int>(window.x().start());
  const auto window_end_x = static_cast<int>(window.x().end());

  // Collapse window and reset first dimension to handle tail calculations manually
  Window win_collapsed = window.collapse_if_possible(window, Window::DimZ);
  win_collapsed.set(Window::DimX, Window::Dimension(0, 1, 1));

  // Create iterators
  Iterator in(input, win_collapsed);
  Iterator out(output, win_collapsed);

#ifdef __aarch64__
  constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_NEAREST_EVEN;
#else  //__aarch64__
  constexpr RoundingPolicy rounding_policy = RoundingPolicy::TO_ZERO;
#endif //__aarch64__
  const auto &qinfo_in = input->info()->quantization_info().uniform();
  const auto &qinfo_out = output->info()->quantization_info().uniform();

  execute_window_loop(
      win_collapsed,
      [&](const Coordinates &) {
        const auto in_ptr = reinterpret_cast<const qasymm8_t *>(in.ptr());

        int x = window_start_x;
        for (; x <= (window_end_x - window_step_x); x += window_step_x)
        {
          using from_vector = typename cast_vector<float>::type;
          const auto vf = wrapper::vloadq(in_ptr + x);
          const auto vin = vdequantize(vf, qinfo_in);
          switch (output->info()->data_type())
          {
            case DataType::U8:
            {
              using to_vector = typename cast_vector<uint8_t>::type;
              const to_vector vout = vcast<to_vector, from_vector>(vin);
              store_result<uint8_t>(reinterpret_cast<uint8_t *>(out.ptr()) + x, vout);
              break;
            }
            case DataType::QASYMM8:
            {
              using to_vector = typename cast_vector<float>::type;
              const auto vf = vcast<to_vector, from_vector>(vin);
              const auto vout = vquantize(vf, qinfo_out);
              store_result<qasymm8_t>(reinterpret_cast<qasymm8_t *>(out.ptr()) + x, vout);
              break;
            }
            case DataType::U32:
            {
              using to_vector = typename cast_vector<uint32_t>::type;
              const to_vector vout = vcast<to_vector, from_vector>(vin);
              store_result<uint32_t>(reinterpret_cast<uint32_t *>(out.ptr()) + x, vout);
              break;
            }
            case DataType::S32:
            {
              using to_vector = typename cast_vector<int32_t>::type;
              const to_vector vout = vcast<to_vector, from_vector>(vin);
              store_result<int32_t>(reinterpret_cast<int32_t *>(out.ptr()) + x, vout);
              break;
            }
            case DataType::F32:
            {
              using to_vector = typename cast_vector<float>::type;
              const to_vector vout = vcast<to_vector, from_vector>(vin);
              store_result<float>(reinterpret_cast<float *>(out.ptr()) + x, vout);
              break;
            }
            default:
              ARM_COMPUTE_ERROR("Unsupported data type.");
          }
        }

        // Compute left-over elements
        for (; x < window_end_x; ++x)
        {
          qasymm8_t qval_in = *(in_ptr + x);
          const auto val = dequantize_qasymm8(qval_in, qinfo_in);

          switch (output->info()->data_type())
          {
            case DataType::U8:
            {
              *(reinterpret_cast<uint8_t *>(out.ptr()) + x) = static_cast<uint8_t>(val);
              break;
            }
            case DataType::QASYMM8:
            {
              const auto qval_out = quantize_qasymm8(val, qinfo_out, rounding_policy);
              *(reinterpret_cast<qasymm8_t *>(out.ptr()) + x) = qval_out;
              break;
            }
            case DataType::U32:
            {
              *(reinterpret_cast<uint32_t *>(out.ptr()) + x) = static_cast<uint32_t>(val);
              break;
            }
            case DataType::S32:
            {
              *(reinterpret_cast<int32_t *>(out.ptr()) + x) = static_cast<int32_t>(val);
              break;
            }
            case DataType::F32:
            {
              *(reinterpret_cast<float *>(out.ptr()) + x) = static_cast<float>(val);
              break;
            }
            default:
              ARM_COMPUTE_ERROR("Unsupported data type.");
          }
        }
      },
      in, out);
}
} // namespace

NECastKernel::NECastKernel() : _input(nullptr), _output(nullptr), _input_subtype(SubDataType::NONE)
{
}

void NECastKernel::configure(const ITensor *input, ITensor *output, SubDataType input_subtype)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input, output);
  ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(input->info(), output->info(), input_subtype));

  _input = input;
  _output = output;
  _input_subtype = input_subtype;

  // Configure kernel window
  auto win_config = validate_and_configure_window(input->info(), output->info());

  ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

  INEKernel::configure(std::get<1>(win_config));
}

Status NECastKernel::validate(const ITensorInfo *input, const ITensorInfo *output,
                              SubDataType input_subtype)
{
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, input_subtype));
  ARM_COMPUTE_RETURN_ON_ERROR(
      std::get<0>(validate_and_configure_window(input->clone().get(), output->clone().get())));
  return Status{};
}

void NECastKernel::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

  switch (_input->info()->data_type())
  {
    case DataType::U8:
      if (_input_subtype == SubDataType::BOOL)
      {
        run_cast<bool>(_input, _output, window);
      }
      else
      {
        run_cast<uint8_t>(_input, _output, window);
      }
      break;
    case DataType::QASYMM8:
      run_cast_qasymm8(_input, _output, window);
      break;
    case DataType::U32:
      run_cast<uint32_t>(_input, _output, window);
      break;
    case DataType::S32:
      run_cast<int32_t>(_input, _output, window);
      break;
    case DataType::F32:
      run_cast<float>(_input, _output, window);
      break;
    default:
      ARM_COMPUTE_ERROR("Unsupported data type.");
  }
}
} // namespace arm_compute
