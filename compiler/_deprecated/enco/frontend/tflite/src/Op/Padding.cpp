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

#include "Padding.h"

#include "Convert.h"
#include "TensorBags.h"

#include <coco/IR/Data.h>
#include <coco/IR/Module.h>

#include <nncc/core/ADT/tensor/Shape.h>
#include <schema_generated.h>

#include <map>
#include <sstream>
#include <algorithm>
#include <cassert>

using namespace nncc::core::ADT;

namespace tflimport
{

coco::Padding2D get_padding(const tensor::Shape &ifm_shape, const int kernel_w, const int kernel_h,
                            tflite::Padding padding, int stride_w, int stride_h,
                            int dilation_w_factor, int dilation_h_factor)
{
  assert(stride_w != 0);
  assert(stride_h != 0);
  assert(ifm_shape.rank() == 4);

  /**
   * Compute [top padding + bottom padding] (or [left padding + right padding]).
   * If this returns an even number, top = return value / 2 and bottom = return value - top
   * If this returns an odd number, top = return value / 2 and bottom = return value - top (so,
   * bottom = top + 1)
   *
   * Code based on https://www.tensorflow.org/api_guides/python/nn#Convolution
   */
  auto compute_padding = [](tflite::Padding padding, int stride, int dilation_rate, int in_size,
                            int filter_size) {
    int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
    if (padding == tflite::Padding_SAME)
    {
      if (in_size % stride == 0)
        return std::max(effective_filter_size - stride, 0);
      else
        return std::max(effective_filter_size - (in_size % stride), 0);
    }
    else // padding == VALID
    {
      return 0;
    }
  };

  // ifm shape is from order of NHWC. ifm W = dim(2), ifm H = dim(1)
  int padding_w = compute_padding(padding, stride_w, dilation_w_factor, ifm_shape.dim(2), kernel_w);
  int padding_h = compute_padding(padding, stride_h, dilation_h_factor, ifm_shape.dim(1), kernel_h);

  coco::Padding2D coco_padding;
  coco_padding.top(padding_h / 2).bottom(padding_h - padding_h / 2);
  coco_padding.left(padding_w / 2).right(padding_w - padding_w / 2);

  return coco_padding;
}

coco::Padding2D pool2D_padding(const tflite::Pool2DOptions *options, const tensor::Shape &ifm_shape,
                               const int filter_w, const int filter_h)
{
  return get_padding(ifm_shape, filter_w, filter_h, options->padding(), options->stride_w(),
                     options->stride_h(), 1, 1);
}

coco::Padding2D conv2D_padding(const tflite::Conv2DOptions *options, const tensor::Shape &ifm_shape,
                               const tensor::Shape &kernel_shape)
{
  return get_padding(ifm_shape, kernel_shape.dim(2), kernel_shape.dim(1), /* kernel layout: NHWC */
                     options->padding(), options->stride_w(), options->stride_h(),
                     options->dilation_w_factor(), options->dilation_h_factor());
}

coco::Padding2D depthwiseConv2D_padding(const tflite::DepthwiseConv2DOptions *options,
                                        const tensor::Shape &ifm_shape,
                                        const tensor::Shape &kernel_shape)
{
  return get_padding(ifm_shape, kernel_shape.dim(2), kernel_shape.dim(1), /* kernel layout: NHWC */
                     options->padding(), options->stride_w(), options->stride_h(),
                     options->dilation_w_factor(), options->dilation_h_factor());
}

} // namespace tflimport
