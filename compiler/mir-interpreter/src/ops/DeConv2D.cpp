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

#include <cstdlib>

#include "mir/TensorUtil.h"
#include "mir/ShapeRange.h"
#include "mir/Tensor.h"

#include "DeConv2D.h"
#include "Common.h"

namespace mir_interpreter
{

using namespace mir;
using namespace mir::ops;

template <typename T> struct DeConv2DImpl
{
  static void run(const mir::TensorVariant &input_var, const mir::TensorVariant &kernel_var,
                  const mir::ops::DeConv2DOp &op, mir::TensorVariant &output);
};

template <typename T>
void DeConv2DImpl<T>::run(const TensorVariant &input_var, const TensorVariant &kernel_var,
                          const DeConv2DOp &op, TensorVariant &output)
{
  Tensor<T> input(input_var);

  // Input shape: [N, Hi, Wi, Ci]
  // Kernel shape: [Hk, Wk, Co, Ci]
  assert(op.getInputShape(0).rank() == 4);
  const auto &input_shape = input.getShape();
  const auto &kernel_shape = kernel_var.getShape();
  (void)input_shape;
  (void)kernel_shape;
  assert(input_shape.rank() == 4);
  assert(kernel_shape.rank() == 4);
  assert(kernel_shape.dim(3) == input_shape.dim(3));
  assert(op.getPaddingBefore().size() == 2);
  assert(op.getPaddingAfter().size() == 2);

  const auto &strides = op.getStrides();
  Shape out_shape = op.getOutputShape(0);
  Tensor<T> res_accesor(output);
  Index pads({op.getPaddingBefore().at(0), op.getPaddingBefore().at(1), 0});

  out_shape.dim(3) = 1;
  ShapeRange out_range(out_shape);

  Shape in_shape = input.getShape();
  ShapeRange in_range(in_shape);

  auto tr_kernel = transposeTensor<0, 1, 3, 2>(kernel_var);
  Tensor<T> kernel(tr_kernel);

  Shape k_shape = kernel.getShape();
  int32_t num_kernels = k_shape.dim(3);
  k_shape.dim(3) = 1;
  ShapeRange kernel_range(k_shape);

  Index input_idx;
  input_idx.resize(in_shape.rank());

  Index kernel_idx;
  kernel_idx.resize(k_shape.rank());

  erase<T>(output);

  for (auto &out_idx : out_range)
  {
    auto out_region = res_accesor.getRegion(out_idx);
    assert(out_region.size() == num_kernels);

    for (auto &kernel_idx_r : kernel_range)
    {
      // rotate kernel 180 deg around last axis
      // by index transform
      for (int32_t d = 0; d < 2; ++d)
      {
        kernel_idx.at(d) = kernel.getShape().dim(d) - kernel_idx_r.at(d) - 1;
      }
      kernel_idx.at(2) = kernel_idx_r.at(2);
      kernel_idx.at(3) = kernel_idx_r.at(3);

      // flag that keeps info on whether the current input element is from input
      // or is from dilation by stride
      bool is_from_input = true;
      for (int32_t d = 1; d < input_idx.rank() - 1; ++d)
      {
        const auto num = (out_idx.at(d) + pads.at(d - 1) - kernel_idx.at(d - 1));
        auto stride = strides[d - 1];
        const auto div_res = num / stride;
        const auto rem = num - div_res * stride;
        is_from_input = is_from_input && rem == 0;
        if (rem != 0)
          break;
        input_idx.at(d) = div_res;
      }
      if (is_from_input)
      {
        // batch is same as output's
        input_idx.at(0) = out_idx.at(0);
        // channel index - same as kernel's
        input_idx.at(3) = kernel_idx.at(2);

        if (in_range.contains(input_idx))
        {
          auto kernel_region = kernel.getRegion(kernel_idx);
          assert(kernel_region.size() == num_kernels);

          auto in = input.at(input_idx);

          for (int32_t kernel_index = 0; kernel_index < num_kernels; kernel_index++)
          {
            out_region.base()[kernel_index] += in * kernel_region.base()[kernel_index];
          }
        }
      }
    }
  }
}

void DeConv2D(const mir::TensorVariant &input, const mir::TensorVariant &kernel,
              const mir::ops::DeConv2DOp &op, mir::TensorVariant &output)
{
  dispatch<DeConv2DImpl>(output.getElementType(), input, kernel, op, output);
}

} // namespace mir_interpreter
