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
 * Copyright (c) 2019 ARM Limited.
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

#include "arm_compute/core/NEON/kernels/NEInstanceNormalizationLayerKernelEx.h"

#include "src/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "src/core/NEON/NEMath.h"
#include "src/core/NEON/INEKernel.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"

#include "src/core/NEON/wrapper/wrapper.h"

#include "src/core/helpers/WindowHelpers.h"
#include "src/core/helpers/AutoConfiguration.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace
{
template <typename T>
void instance_normalization_nchw(ITensor *input, ITensor *output, ITensor *gamma, ITensor *beta,
                                 float epsilon, const Window &window)
{
  /** NEON vector tag type. */
  using ExactTagType =
    typename wrapper::traits::neon_bitvector_tag_t<T, wrapper::traits::BitWidth::W128>;

  // Clear X/Y dimensions on execution window as we handle the planes manually
  Window win = window;
  win.set(Window::DimX, Window::Dimension(0, 1, 1));
  win.set(Window::DimY, Window::Dimension(0, 1, 1));

  constexpr int window_step_x = 16 / sizeof(T);
  const unsigned int elements_plane = input->info()->dimension(0) * output->info()->dimension(1);
  const auto channel_idx =
    get_data_layout_dimension_index(input->info()->data_layout(), DataLayoutDimension::CHANNEL);

  Iterator input_it(input, win);
  execute_window_loop(
    win,
    [&](const Coordinates &id) {
      Window win_plane = window;
      win_plane.set(Window::DimX, Window::Dimension(0, 1, 1));
      win_plane.set(Window::DimZ, Window::Dimension(id[2], id[2] + 1, 1));
      win_plane.set(3, Window::Dimension(id[3], id[3] + 1, 1));

      Iterator input_plane_it(input, win_plane);
      Iterator output_plane_it(output, win_plane);

      auto sum_h_w = static_cast<T>(0.f);
      auto sum_squares_h_w = static_cast<T>(0.f);

      execute_window_loop(
        win_plane,
        [&](const Coordinates &) {
          const auto input_ptr = reinterpret_cast<const T *>(input_plane_it.ptr());

          auto vec_sum_h_w = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});
          auto vec_sum_squares_h_w = wrapper::vdup_n(static_cast<T>(0.f), ExactTagType{});

          // Compute S elements per iteration
          int x = window.x().start();
          for (; x <= (window.x().end() - window_step_x); x += window_step_x)
          {
            auto vec_input_val = wrapper::vloadq(input_ptr + x);
            vec_sum_h_w = wrapper::vadd(vec_sum_h_w, vec_input_val);
            vec_sum_squares_h_w =
              wrapper::vadd(vec_sum_squares_h_w, wrapper::vmul(vec_input_val, vec_input_val));
          }

          auto vec2_sum_h_w =
            wrapper::vpadd(wrapper::vgethigh(vec_sum_h_w), wrapper::vgetlow(vec_sum_h_w));
          auto vec2_sum_squares_h_w = wrapper::vpadd(wrapper::vgethigh(vec_sum_squares_h_w),
                                                     wrapper::vgetlow(vec_sum_squares_h_w));
          for (int i = 0; i < window_step_x / 4; ++i)
          {
            vec2_sum_h_w = wrapper::vpadd(vec2_sum_h_w, vec2_sum_h_w);
            vec2_sum_squares_h_w = wrapper::vpadd(vec2_sum_squares_h_w, vec2_sum_squares_h_w);
          }
          sum_h_w += wrapper::vgetlane(vec2_sum_h_w, 0);
          sum_squares_h_w += wrapper::vgetlane(vec2_sum_squares_h_w, 0);

          // Compute left-over elements
          for (; x < window.x().end(); ++x)
          {
            const auto value = *(input_ptr + x);
            sum_h_w += value;
            sum_squares_h_w += value * value;
          }
        },
        input_plane_it, output_plane_it);

      const auto mean_h_w = sum_h_w / elements_plane;
      const auto var_h_w = sum_squares_h_w / elements_plane - mean_h_w * mean_h_w;

      auto gamma_val = 1.0f;
      if (gamma != nullptr)
      {
        gamma_val = *reinterpret_cast<T *>(gamma->ptr_to_element({id[channel_idx]}));
      }
      const auto multip_h_w = gamma_val / std::sqrt(var_h_w + epsilon);
      const auto vec_mean_h_w = wrapper::vdup_n(static_cast<T>(mean_h_w), ExactTagType{});
      const auto vec_multip_h_w = wrapper::vdup_n(static_cast<T>(multip_h_w), ExactTagType{});
      auto beta_val = 0.0f;
      if (beta != nullptr)
      {
        beta_val = *reinterpret_cast<T *>(beta->ptr_to_element({id[channel_idx]}));
      }
      const auto vec_beta = wrapper::vdup_n(static_cast<T>(beta_val), ExactTagType{});

      execute_window_loop(
        win_plane,
        [&](const Coordinates &) {
          auto input_ptr = reinterpret_cast<T *>(input_plane_it.ptr());
          auto output_ptr = reinterpret_cast<T *>(output_plane_it.ptr());

          // Compute S elements per iteration
          int x = window.x().start();
          auto vec_val = wrapper::vdup_n(static_cast<T>(0.0f), ExactTagType{});
          for (; x <= (window.x().end() - window_step_x); x += window_step_x)
          {
            vec_val = wrapper::vloadq(input_ptr + x);
            vec_val = wrapper::vadd(
              wrapper::vmul(wrapper::vsub(vec_val, vec_mean_h_w), vec_multip_h_w), vec_beta);
            wrapper::vstore(output_ptr + x, vec_val);
          }

          // Compute left-over elements
          for (; x < window.x().end(); ++x)
          {
            *(output_ptr + x) = ((*(input_ptr + x)) - mean_h_w) * multip_h_w + beta_val;
          }
        },
        input_plane_it, output_plane_it);
    },
    input_it);
}

Status validate_arguments(const ITensorInfo *input, const ITensorInfo *output,
                          const ITensorInfo *gamma, const ITensorInfo *beta, float epsilon)
{
  ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(input);
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(epsilon == 0.f, "Epsilon must be different than 0");

  ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(input, DataType::F16, DataType::F32);
  ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->data_layout() == DataLayout::NHWC,
                                  "NHWC data layout is not supported by the kernel directly");

  if (output != nullptr && output->total_size() != 0)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_LAYOUT(input, output);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->num_channels() != output->num_channels(),
                                    "Input and output have different number of channels");
  }

  if (gamma != nullptr)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, gamma);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->dimension(get_data_layout_dimension_index(
                                      input->data_layout(), DataLayoutDimension::CHANNEL)) !=
                                      gamma->dimension(0),
                                    "Gamma's size must be the same as size of input's channel");
  }

  if (beta != nullptr)
  {
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(input, beta);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(input->dimension(get_data_layout_dimension_index(
                                      input->data_layout(), DataLayoutDimension::CHANNEL)) !=
                                      beta->dimension(0),
                                    "Beta's size must be the same as size of input's channel");
  }

  return Status{};
}

std::tuple<Status, Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)
{
  // We handle the planes manually
  Window win = calculate_max_window(*input, Steps(1));

  // Output auto initialization if not yet initialized
  auto_init_if_empty(*output, input->tensor_shape(), 1, input->data_type());

  // NEInstanceNormalizationLayerKernelEx doesn't need padding so update_window_and_padding() can be
  // skipped
  Coordinates coord;
  coord.set_num_dimensions(output->num_dimensions());
  output->set_valid_region(ValidRegion(coord, output->tensor_shape()));
  return std::make_pair(Status{}, win);
}
} // namespace

NEInstanceNormalizationLayerKernelEx::NEInstanceNormalizationLayerKernelEx()
  : _func(nullptr), _input(nullptr), _output(nullptr), _gamma(nullptr), _beta(nullptr),
    _epsilon(1e-12)
{
}

void NEInstanceNormalizationLayerKernelEx::configure(ITensor *input, ITensor *output,
                                                     ITensor *gamma, ITensor *beta, float epsilon)
{
  ARM_COMPUTE_ERROR_ON_NULLPTR(input);

  _input = input;
  _output = output == nullptr ? input : output;
  _gamma = gamma;
  _beta = beta;
  _epsilon = epsilon;

  ARM_COMPUTE_ERROR_THROW_ON(
    validate_arguments(_input->info(), _output->info(), gamma->info(), beta->info(), epsilon));

  if (_input->info()->data_type() == DataType::F32)
  {
    _func = &instance_normalization_nchw<float>;
  }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  else if (_input->info()->data_type() == DataType::F16)
  {
    _func = &instance_normalization_nchw<float16_t>;
  }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
  else
  {
    ARM_COMPUTE_ERROR("Unsupported data type");
  }

  // Configure kernel window
  auto win_config = validate_and_configure_window(_input->info(), _output->info());
  ARM_COMPUTE_ERROR_THROW_ON(std::get<0>(win_config));

  INEKernel::configure(std::get<1>(win_config));
}

Status NEInstanceNormalizationLayerKernelEx::validate(const ITensorInfo *input,
                                                      const ITensorInfo *output,
                                                      const ITensorInfo *gamma,
                                                      const ITensorInfo *beta, float epsilon)
{
  ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, output, gamma, beta, epsilon));
  ARM_COMPUTE_RETURN_ON_ERROR(std::get<0>(validate_and_configure_window(
    input->clone().get(), (output == nullptr ? input->clone().get() : output->clone().get()))));
  return Status{};
}

void NEInstanceNormalizationLayerKernelEx::run(const Window &window, const ThreadInfo &info)
{
  ARM_COMPUTE_UNUSED(info);
  ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
  ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);
  (*_func)(_input, _output, _gamma, _beta, _epsilon, window);
}
} // namespace arm_compute
