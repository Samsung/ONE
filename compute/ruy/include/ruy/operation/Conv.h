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

#ifndef __NNFW_RUY_CONV_H__
#define __NNFW_RUY_CONV_H__

#include "ruy/Types.h"
#include "ruy/Shape.h"
#include "ruy/Utils.h"
#include "ruy/RuySupport.h"

#include <ruy/ruy.h>
#include <ruy/context.h>
#include <iostream>
#include <vector>

namespace nnfw
{
namespace ruy
{

class Conv
{
public:
  Conv() : _im2col_shape(4), _need_im2col(false), _prepared(false) {}

  void prepare(const Shape &input_shape, const Shape &kernel_shape, const Shape &output_shape,
               uint32_t stride_width, uint32_t stride_height, uint32_t dilation_width_factor,
               uint32_t dilation_height_factor)
  {
    if (!_prepared)
    {
      IsRequiredIm2col(input_shape, kernel_shape, output_shape, stride_width, stride_height,
                       dilation_width_factor, dilation_height_factor);
      _prepared = true;
    }
  }

  void operator()(const ConvParams &params, const Shape &input_shape, const float *input_data,
                  const Shape &filter_shape, const float *filter_data, const Shape &bias_shape,
                  const float *bias_data, const Shape &output_shape, float *output_data,
                  ::ruy::Context *ruy_context)
  {
    if (!_prepared)
    {
      // This means that input or output are dynamic or filter is not constant
      IsRequiredIm2col(input_shape, filter_shape, output_shape, params.stride_width,
                       params.stride_height, params.dilation_width_factor,
                       params.dilation_height_factor);
      _prepared = true;
    }

    int im2col_size = _need_im2col ? _im2col_shape.FlatSize() : 0;

    // Use heap if size is larger than 8MB
    if (im2col_size > 2 * 1024 * 1024)
    {
      std::unique_ptr<float[]> im2col_data = std::make_unique<float[]>(im2col_size);
      ConvFloat(params, input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
                output_shape, output_data, _im2col_shape, im2col_data.get(), ruy_context);
    }
    else if (im2col_size > 0)
    {
      float im2col_data[im2col_size];
      ConvFloat(params, input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
                output_shape, output_data, _im2col_shape, im2col_data, ruy_context);
    }
    else
    {
      ConvFloat(params, input_shape, input_data, filter_shape, filter_data, bias_shape, bias_data,
                output_shape, output_data, _im2col_shape, nullptr, ruy_context);
    }
  }

private:
  void ConvFloat(const ConvParams &params, const Shape &input_shape, const float *input_data,
                 const Shape &filter_shape, const float *filter_data, const Shape &bias_shape,
                 const float *bias_data, const Shape &output_shape, float *output_data,
                 const Shape &im2col_shape, float *im2col_data, ::ruy::Context *ruy_context)
  {
    UNUSED_RELEASE(bias_shape);
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int dilation_width_factor = params.dilation_width_factor;
    const int dilation_height_factor = params.dilation_height_factor;
    const float output_activation_min = params.float_activation_min;
    const float output_activation_max = params.float_activation_max;
    assert(input_shape.DimensionsCount() == 4);
    assert(filter_shape.DimensionsCount() == 4);
    assert(output_shape.DimensionsCount() == 4);

    // NB: the float 0.0f value is represented by all zero bytes.
    const uint8_t float_zero_byte = 0x00;
    const float *gemm_input_data = nullptr;
    const Shape *gemm_input_shape = nullptr;
    const int filter_width = filter_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const bool need_dilated_im2col = dilation_width_factor != 1 || dilation_height_factor != 1;
    const bool need_im2col =
        stride_width != 1 || stride_height != 1 || filter_width != 1 || filter_height != 1;
    if (need_dilated_im2col)
    {
      DilatedIm2col(params, float_zero_byte, input_shape, input_data, filter_shape, output_shape,
                    im2col_data);
      gemm_input_data = im2col_data;
      gemm_input_shape = &im2col_shape;
    }
    else if (need_im2col)
    {
      assert(im2col_data);
      Im2col(params, filter_height, filter_width, float_zero_byte, input_shape, input_data,
             im2col_shape, im2col_data);
      gemm_input_data = im2col_data;
      gemm_input_shape = &im2col_shape;
    }
    else
    {
      // TODO(aselle): We need to make sure to not send im2col if it is not
      // needed.
      assert(!im2col_data);
      gemm_input_data = input_data;
      gemm_input_shape = &input_shape;
    }

    const int gemm_input_dims = gemm_input_shape->DimensionsCount();
    int m = FlatSizeSkipDim(*gemm_input_shape, gemm_input_dims - 1);
    int n = output_shape.Dims(3);
    int k = gemm_input_shape->Dims(gemm_input_dims - 1);

    // When an optimized CBLAS implementation is not available, fall back
    // to using cpu_backend_gemm.
    MatrixParams<float> lhs_params;
    lhs_params.order = Order::kRowMajor;
    lhs_params.rows = n;
    lhs_params.cols = k;
    MatrixParams<float> rhs_params;
    rhs_params.order = Order::kColMajor;
    rhs_params.rows = k;
    rhs_params.cols = m;
    MatrixParams<float> dst_params;
    dst_params.order = Order::kColMajor;
    dst_params.rows = n;
    dst_params.cols = m;
    GemmParams<float, float> gemm_params;
    gemm_params.bias = bias_data;
    gemm_params.clamp_min = output_activation_min;
    gemm_params.clamp_max = output_activation_max;

    // Below code is from tflite::cpu_backend_gemm::detail::GemmImplUsingRuy
    ::ruy::Matrix<float> ruy_lhs;
    ::ruy::Matrix<float> ruy_rhs;
    ::ruy::Matrix<float> ruy_dst;
    // Note that cache is always enabled for input and weight tensors
    ruy_support::MakeRuyMatrix(lhs_params, filter_data, &ruy_lhs, true);
    ruy_support::MakeRuyMatrix(rhs_params, gemm_input_data, &ruy_rhs, true);
    ruy_support::MakeRuyMatrix(dst_params, output_data, &ruy_dst);

    ::ruy::BasicSpec<float, float> ruy_mul_params;
    ruy_support::MakeRuyMulParams(gemm_params, &ruy_mul_params);

    ::ruy::Mul(ruy_lhs, ruy_rhs, ruy_mul_params, ruy_context, &ruy_dst);
  }

  void IsRequiredIm2col(const Shape &input_shape, const Shape &kernel_shape,
                        const Shape &output_shape, uint32_t stride_width, uint32_t stride_height,
                        uint32_t dilation_width_factor, uint32_t dilation_height_factor)
  {
    const bool need_dilated_im2col = dilation_width_factor != 1 || dilation_height_factor != 1;
    const bool need_non_dilated_im2col = stride_width != 1 || stride_height != 1 ||
                                         kernel_shape.Dims(1) != 1 || kernel_shape.Dims(2) != 1;

    _need_im2col = need_dilated_im2col || need_non_dilated_im2col;

    if (_need_im2col)
    {
      _im2col_shape.SetDim(0, output_shape.Dims(0));
      _im2col_shape.SetDim(1, output_shape.Dims(1));
      _im2col_shape.SetDim(2, output_shape.Dims(2));
      _im2col_shape.SetDim(3, input_shape.Dims(3) * kernel_shape.Dims(1) * kernel_shape.Dims(2));
    }
  }

private:
  Shape _im2col_shape;
  bool _need_im2col;
  bool _prepared;
};
} // namespace ruy
} // namespace nnfw

#endif // __NNFW_RUY_CONV_H_
