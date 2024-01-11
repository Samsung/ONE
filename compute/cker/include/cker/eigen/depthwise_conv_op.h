/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_EIGEN_DEPTHWISE_CONV_OP_H__
#define __NNFW_CKER_EIGEN_DEPTHWISE_CONV_OP_H__

// From tensorflow/core/kernels/depthwise_conv_grad_op.cc
#define EIGEN_USE_THREADS

#include <thread>
#include "unsupported/Eigen/CXX11/Tensor"
#include "cker/operation/Helper/Tensor.h"

// From tensorflow/core/kernels/depthwise_conv_op.h
namespace nnfw
{
namespace cker
{
namespace depthwise_conv_op
{

template <typename Device, typename T> struct LaunchDepthwiseConvBackpropInputOp
{
  void operator()(int batch, int in_rows, int in_cols, int in_depth, int filter_rows,
                  int filter_cols, int depth_multiplier, int stride, int pad_rows, int pad_cols,
                  int out_rows, int out_cols, int out_depth, const T *out_backprop, const T *filter,
                  T *in_backprop);
};

template <typename Device, typename T> struct LaunchDepthwiseConvBackpropFilterOp
{
  void operator()(int batch, int in_rows, int in_cols, int in_depth, int filter_rows,
                  int filter_cols, int depth_multiplier, int stride, int pad_rows, int pad_cols,
                  int out_rows, int out_cols, int out_depth, const T *out_backprop, const T *input,
                  T *filter_backprop);
};

namespace functor
{

// Pads 'filter' to vector-register boundary along its inner dimension:
//   filter_inner_dim_size = in_depth * depth_multiplier
// Requires 'filter' to have the following storage order:
//   [filter_rows, filter_cols, in_depth, depth_multiplier]
// Returns zero-padded filter in 'padded_filter'.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   So we have a total of 3 * 2 = 6 filters, each of spatial size 2 x 2.
//
//   filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, u1, v1] [w1, x1, y1, z1]
//     [u2, v2, w2, x2] [y2, z2, u3, v3] [w3, x3, y3, z3]
//
//   padded_filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, 0, 0] [u1, v1, w1, x1] [y1, z1, 0, 0]
//     [u2, v2, w2, x2] [y2, z2, 0, 0] [u3, v3, w3, x3] [y3, z3, 0, 0]

template <typename T> struct DepthwiseFilterPadOp
{
  void operator()(int, int, int, int, int filter_rows, int filter_cols, int, int, int, int, int,
                  int, int out_depth, const T *filter, T *padded_filter)
  {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));

    // Calculate vectorized and scalar lengths of filter's inner dimension.
    const int64_t filter_inner_dim_size = out_depth;
    const int64_t vectorized_size = (filter_inner_dim_size / kPacketSize) * kPacketSize;
    const int64_t scalar_size = filter_inner_dim_size - vectorized_size;
    // Calculate required padding and padded output buffer stride.
    const int64_t pad_size = scalar_size > 0 ? kPacketSize - scalar_size : 0;
    const int64_t padded_filter_stride = vectorized_size + kPacketSize;

    const int64_t filter_spatial_size = filter_rows * filter_cols;
    for (int64_t i = 0; i < filter_spatial_size; ++i)
    {
      const int64_t input_base = i * filter_inner_dim_size;
      const int64_t output_base = i * padded_filter_stride;
      // Write vectorized length of filter's inner dimension to output.
      for (int64_t j = 0; j < vectorized_size; j += kPacketSize)
      {
        const auto v = Eigen::internal::ploadu<Packet>(filter + input_base + j);
        Eigen::internal::pstoreu<T>(padded_filter + output_base + j, v);
      }
      // Write scalar length of filter's inner dimension to output.
      for (int64_t j = 0; j < scalar_size; ++j)
      {
        padded_filter[output_base + vectorized_size + j] = filter[input_base + vectorized_size + j];
      }
      // Pad the remainder of output to vector-register boundary.
      for (int64_t j = 0; j < pad_size; ++j)
      {
        padded_filter[output_base + vectorized_size + scalar_size + j] = static_cast<T>(0);
      }
    }
  }
};

// Copies data from local region in 'input' specified by 'out_r' and 'out_'c'
// to 'input_buffer'. The copied data is replicated by factor
// 'depth_multiplier', and padded to vector register-width boundaries so
// that it is aligned for efficient traversal and vector multiply-add by the
// depthwise kernel.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//
//   input: [batch, in_rows, in_cols, in_depth]
//
//     [a0, a1, a2, b0, b1, b2, ..., e0, e1, e2, f0, f1, f2, ...]
//
//   input_buffer (register boundaries shown):
//     [a0, a0, a1, a1] [a2, a2, 0, 0]   in_row = 0, in_col = 0
//     [b0, b0, b1, b1] [b2, b2, 0, 0]   in_row = 0, in_col = 1
//     [e0, e0, e1, e1] [e2, e2, 0, 0]   in_row = 1, in_col = 0
//     [f0, f0, f1, f1] [f2, f2, 0, 0]   in_row = 1, in_col = 1
//
// Returns replicated and padded data from specified input region in
// 'input_buffer'.

template <typename T> struct DepthwiseInputCopyOp
{
  void operator()(int, int in_rows, int in_cols, int in_depth, int filter_rows, int filter_cols,
                  int depth_multiplier, int stride, int pad_rows, int pad_cols, int, int,
                  int out_depth, const int64_t padded_filter_inner_dim_size, const int64_t out_r,
                  const int64_t out_c, const T *input, T *input_buffer)
  {
    typedef typename Eigen::internal::packet_traits<T>::type Packet;
    static const int64_t kPacketSize = Eigen::internal::packet_traits<T>::size;

    const int64_t kDepth = depth_multiplier;
    // Calculate vectorized and scalar (residual) lengths for 'in_depth'.
    const int64_t input_vectorized_size = (in_depth / kPacketSize) * kPacketSize;
    const int64_t input_scalar_size = in_depth - input_vectorized_size;

    // Calculate output padding length.
    const int64_t output_scalar_size = out_depth % kPacketSize;
    const int64_t output_pad_size = output_scalar_size > 0 ? kPacketSize - output_scalar_size : 0;

    // Iterate through all rows x cols reading 'in_depth' from 'input' and
    // replicating by 'depth_multiplier' into 'input_buffer' (otherwise
    // zero-padding input buffer as needed).
    auto *in_buf = input_buffer;
    const int64_t in_r_start = out_r * stride - pad_rows;
    const int64_t in_c_start = out_c * stride - pad_cols;

    // TODO: add a ploaddup variant for depth == 2 if needed.
    if (kDepth > 1 && kDepth <= kPacketSize)
    {
      for (int64_t f_r = 0; f_r < filter_rows; ++f_r)
      {
        const int64_t in_r = in_r_start + f_r;

        for (int64_t f_c = 0; f_c < filter_cols; ++f_c)
        {
          const int64_t in_c = in_c_start + f_c;

          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols)
          {
            const auto *in = input + (in_r * in_cols + in_c) * in_depth;
            int64_t limit = in_depth;
            // This will overwrite up to kPacketSize next elements,
            // this is ok on all iterations except the last one, since
            // we will write correct values on a next iteration.
            if (f_c == filter_cols - 1)
            {
              limit -= (kPacketSize - kDepth) / kDepth + 1;
              if (limit < 0)
              {
                limit = 0;
              }
            }
            // Copy vectorized portion of inner dimension.
            for (int64_t d = 0; d < limit; d++)
            {
              const auto p = Eigen::internal::pset1<Packet>(in[d]);
              Eigen::internal::pstoreu<T>(in_buf, p);
              in_buf += kDepth;
            }

            // Copy the scalar portion.
            for (int64_t d = limit; d < in_depth; d++)
            {
              const auto value = in[d];
              for (int64_t dm = 0; dm < kDepth; dm++)
              {
                in_buf[dm] = value;
              }
              in_buf += kDepth;
            }

            // Pad the remainder of the output to vector register boundary.
            for (int64_t d = 0; d < output_pad_size; ++d)
            {
              in_buf[d] = static_cast<T>(0);
            }
            in_buf += output_pad_size;
          }
          else
          {
            // Zero pad.
            memset(in_buf, 0, sizeof(T) * padded_filter_inner_dim_size);
            in_buf += padded_filter_inner_dim_size;
          }
        }
      }
    }
    else if (kDepth > kPacketSize)
    {
      // Calculate vectorized and scalar (residual) lengths for
      // 'depth_multiplier'. This is used to efficiently replicate data for
      // when 'depth_multiplier' > kPacketSize.
      const int64_t dm_vectorized_size = (kDepth / kPacketSize) * kPacketSize;

      for (int64_t f_r = 0; f_r < filter_rows; ++f_r)
      {
        const int64_t in_r = in_r_start + f_r;

        for (int64_t f_c = 0; f_c < filter_cols; ++f_c)
        {
          const int64_t in_c = in_c_start + f_c;

          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols)
          {
            const auto *in = input + (in_r * in_cols + in_c) * in_depth;
            // Copy vectorized portion of inner dimension.
            for (int64_t d = 0; d < in_depth; d++)
            {
              const auto p = Eigen::internal::pset1<Packet>(in[d]);
              for (int64_t dm = 0; dm < dm_vectorized_size; dm += kPacketSize)
              {
                Eigen::internal::pstoreu<T>(in_buf + dm, p);
              }
              // Overlapping store for the remainder.
              Eigen::internal::pstoreu<T>(in_buf + kDepth - kPacketSize, p);
              in_buf += kDepth;
            }
            // Pad the remainder of the output to vector register boundary.
            for (int64_t d = 0; d < output_pad_size; ++d)
            {
              in_buf[d] = static_cast<T>(0);
            }
            in_buf += output_pad_size;
          }
          else
          {
            // Zero pad.
            memset(in_buf, 0, sizeof(T) * padded_filter_inner_dim_size);
            in_buf += padded_filter_inner_dim_size;
          }
        }
      }
    }
    else if (kDepth == 1)
    {
      for (int64_t f_r = 0; f_r < filter_rows; ++f_r)
      {
        const int64_t in_r = in_r_start + f_r;

        for (int64_t f_c = 0; f_c < filter_cols; ++f_c)
        {
          const int64_t in_c = in_c_start + f_c;

          if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols)
          {
            const auto *in = input + (in_r * in_cols + in_c) * in_depth;
            for (int64_t d = 0; d < input_vectorized_size; d += kPacketSize)
            {
              const auto p = Eigen::internal::ploadu<Packet>(in + d);
              Eigen::internal::pstoreu<T>(in_buf, p);
              in_buf += kPacketSize;
            }
            for (int64_t d = 0; d < input_scalar_size; ++d)
            {
              T v = in[input_vectorized_size + d];
              in_buf[d] = v;
            }
            in_buf += input_scalar_size;

            // Pad the remainder of the output to vector register boundary.
            for (int64_t d = 0; d < output_pad_size; ++d)
            {
              in_buf[d] = static_cast<T>(0);
            }
            in_buf += output_pad_size;
          }
          else
          {
            // Zero pad.
            memset(in_buf, 0, sizeof(T) * padded_filter_inner_dim_size);
            in_buf += padded_filter_inner_dim_size;
          }
        }
      }
    }
  }
};

} // namespace functor
} // namespace depthwise_conv_op
} // namespace cker
} // namespace nnfw

// From tensorflow/core/kernels/depthwise_conv_grad_op.cc
namespace nnfw
{
namespace cker
{
namespace depthwise_conv_op
{

// Enable CPUDevice only for depthwise_conv_op
using CPUDevice = Eigen::ThreadPoolDevice;

// Copies data from local region in 'out_backprop' into 'buffer'.
// The local region coordinates are calculated as the set of output points which
// used the input point ('in_r', 'in_'c') as input during the forward pass.
// Rather than spatially reversing the filter, the input is reversed during
// the copy. The copied data is padded to vector register-width boundaries so
// that it is aligned for efficient traversal and vector multiply-add by the
// depthwise input kernel.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//
//   'out_backprop': [batch, out_rows, out_cols, out_depth]
//
//     [a00, a01, a10, a11] [a20, a21, b00, b01]
//     [b10, b11, b20, b21] [...]
//     [e00, e01, e10, e11] [e20, e21, f00, f01]
//     [f10, f11, f20, f21] [...]
//
//   'buffer' (register boundaries shown):
//
//     [f00, f01, f10, f11] [f20, f21, 0, 0]   in_row = 0, in_col = 0
//     [e00, e01, e10, e11] [e20, e21, 0, 0]   in_row = 0, in_col = 1
//     [b00, b01, b10, b11] [b20, b21, 0, 0]   in_row = 1, in_col = 0
//     [a00, a01, a10, a11] [a20, a21, 0, 0]   in_row = 1, in_col = 1
//
template <typename T>
static void CopyOutputBackpropRegion(int, int, int, int, int filter_rows_, int filter_cols_, int,
                                     int stride_, int pad_rows_, int pad_cols_, int out_rows_,
                                     int out_cols_, int out_depth,
                                     const int64_t padded_filter_inner_dim_size, const int64_t in_r,
                                     const int64_t in_c, const T *out_backprop, T *buffer)
{
  typedef typename Eigen::internal::packet_traits<T>::type Packet;
  static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));

  const int64_t stride = stride_;
  const int64_t filter_rows = filter_rows_;
  const int64_t filter_cols = filter_cols_;
  const int64_t pad_rows = pad_rows_;
  const int64_t pad_cols = pad_cols_;
  const int64_t out_rows = out_rows_;
  const int64_t out_cols = out_cols_;

  // Calculate the output spatial region which used point (in_r, in_c) as input.
  const int64_t out_r_start =
    std::max(static_cast<int64_t>(0), (in_r - filter_rows + pad_rows + stride) / stride);
  const int64_t out_r_end = std::min(out_rows - 1, (in_r + pad_rows) / stride);
  const int64_t out_c_start =
    std::max(static_cast<int64_t>(0), (in_c - filter_cols + pad_cols + stride) / stride);
  const int64_t out_c_end = std::min(out_cols - 1, (in_c + pad_cols) / stride);

  // Zero-pad 'buffer' if output region is smaller than filter spatial size.
  const int64_t filter_spatial_size = filter_rows * filter_cols;
  if ((out_r_end - out_r_start + 1) < filter_rows || (out_c_end - out_c_start + 1) < filter_cols)
  {
    memset(buffer, 0, filter_spatial_size * padded_filter_inner_dim_size * sizeof(T));
  }

  // Calculate vectorized and scalar (residual) lengths for 'in_depth'.
  const int64_t vectorized_size = (out_depth / kPacketSize) * kPacketSize;
  const int64_t scalar_size = out_depth % kPacketSize;
  const int64_t pad_size = scalar_size > 0 ? kPacketSize - scalar_size : 0;

  for (int out_r = out_r_start; out_r <= out_r_end; ++out_r)
  {
    const int64_t f_r = in_r + pad_rows - out_r * stride;
    for (int out_c = out_c_start; out_c <= out_c_end; ++out_c)
    {
      const int64_t f_c = in_c + pad_cols - out_c * stride;
      const int64_t buf_base = (f_r * filter_cols + f_c) * padded_filter_inner_dim_size;
      // Calculate index into 'out_backprop' for coordinate (out_r, out_c).
      auto *out_bprop = out_backprop + (out_r * out_cols + out_c) * out_depth;

      // Copy vectorized portion of inner dimension into 'buffer'.
      for (int64_t d = 0; d < vectorized_size; d += kPacketSize)
      {
        auto v = Eigen::internal::ploadu<Packet>(out_bprop + d);
        Eigen::internal::pstoreu<T>(buffer + buf_base + d, v);
      }
      // Copy scalar portion of out_bprop to 'buffer'
      for (int64_t d = 0; d < scalar_size; ++d)
      {
        buffer[buf_base + vectorized_size + d] = out_bprop[vectorized_size + d];
      }
      // Pad to vector-register width (if needed).
      for (int64_t d = 0; d < pad_size; ++d)
      {
        buffer[buf_base + vectorized_size + scalar_size + d] = static_cast<T>(0);
      }
    }
  }
}

// Computes the vectorized product of 'buffer' and 'filter' and stores
// result in 'output' at location computed from 'in_r' and 'in_c'.
// If depth_multiplier is > 1, the intermediate output is reduced along
// the depth_multiplier dimension.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   Both 'input_buffer' and 'filter' are padded to register-width boundaries.
//
//   'buffer' [rows, cols, in_depth, depth_multiplier]
//
//     [f00, f01, f10, f11] [f20, f21, 0, 0]   in_row = 0, in_col = 0
//     [e00, e01, e10, e11] [e20, e21, 0, 0]   in_row = 0, in_col = 1
//     [b00, b01, b10, b11] [b20, b21, 0, 0]   in_row = 1, in_col = 0
//     [a00, a01, a10, a11] [a20, a21, 0, 0]   in_row = 1, in_col = 1
//
//   filter [rows, cols, in_depth, depth_multiplier]
//     [u0, v0, w0, x0] [y0, z0, 0, 0] [u1, v1, w1, x1] [y1, z1, 0, 0]
//     [u2, v2, w2, x2] [y2, z2, 0, 0] [u3, v3, w3, x3] [y3, z3, 0, 0]
//
//   First output register [in_depth, depth_multiplier]
//     [q00, q01, q10, q11] = ([f00, f01, f10, f11] x [u0, v0, w0, x0]) +
//                            ([e00, e01, e10, e11] x [u1, v1, w1, x1]) +
//                            ([b00, b01, b10, b11] x [u2, v2, w2, x2]) +
//                            ([a00, a01, a10, a11] x [u3, v3, w3, x3])
//
//   Reduction step along depth-multiplier dimension:
//
//     [q00, q01, q10, q11] [q20, q21, 0, 0] -> [r0, r1, r2, 0]
//

template <typename T>
static void ComputeBackpropInput(int, int, int in_cols, int in_depth_, int filter_rows,
                                 int filter_cols, int depth_multiplier_, int, int, int, int, int,
                                 int out_depth_, const int64_t padded_filter_inner_dim_size,
                                 const int64_t in_r, const int64_t in_c, const T *filter,
                                 const T *buffer, T *out_buffer, T *output)
{
  typedef typename Eigen::internal::packet_traits<T>::type Packet;
  static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));

  const int64_t in_depth = in_depth_;
  const int64_t depth_multiplier = depth_multiplier_;
  const int64_t out_depth = out_depth_;
  const int64_t filter_spatial_size = filter_rows * filter_cols;

  // Calculate vectorized and scalar lengths of 'out_depth'.
  const int64_t output_vectorized_size = (out_depth / kPacketSize) * kPacketSize;
  const int64_t output_scalar_size = out_depth % kPacketSize;

  // Calculate base index at which to begin writing output.
  const int64_t base_output_index = (in_r * in_cols + in_c) * in_depth;

  // Calculate vectorized and scalar lengths for 'depth_multiplier'. This is
  // used to efficiently reduce output when 'depth_multiplier' > kPacketSize.
  const int64_t dm_vectorized_size = (depth_multiplier / kPacketSize) * kPacketSize;
  const int64_t dm_scalar_size = depth_multiplier % kPacketSize;

  for (int i = 0; i < output_vectorized_size; i += kPacketSize)
  {
    // Reset accumulator.
    auto vaccum = Eigen::internal::pset1<Packet>(static_cast<T>(0));
    for (int j = 0; j < filter_spatial_size; ++j)
    {
      // Calculate index.
      const int64_t index = i + j * padded_filter_inner_dim_size;
      // Load filter.
      const auto filter_block = Eigen::internal::ploadu<Packet>(filter + index);
      // Load input.
      const auto data_block = Eigen::internal::ploadu<Packet>(buffer + index);
      // Vector multiply-add.
      vaccum = Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
    }
    if (depth_multiplier == 1)
    {
      // Write directly to the output.
      Eigen::internal::pstoreu<T>(output + base_output_index + i, vaccum);
    }
    else
    {
      // Buffer output for subsequent reduction step.
      Eigen::internal::pstoreu<T>(out_buffer + i, vaccum);
    }
  }

  if (output_scalar_size > 0)
  {
    auto vaccum = Eigen::internal::pset1<Packet>(static_cast<T>(0));
    for (int j = 0; j < filter_spatial_size; ++j)
    {
      const int64_t index = output_vectorized_size + j * padded_filter_inner_dim_size;
      const auto filter_block = Eigen::internal::ploadu<Packet>(filter + index);
      const auto data_block = Eigen::internal::ploadu<Packet>(buffer + index);
      vaccum = Eigen::internal::pmadd<Packet>(filter_block, data_block, vaccum);
    }
    // Load accumulator into an array and loop through output.
    T out_buf[kPacketSize];
    Eigen::internal::pstoreu<T>(out_buf, vaccum);
    if (depth_multiplier == 1)
    {
      // Write directly to the output.
      for (int j = 0; j < output_scalar_size; ++j)
      {
        output[base_output_index + output_vectorized_size + j] = out_buf[j];
      }
    }
    else
    {
      // Buffer output for subsequent reduction step.
      for (int j = 0; j < output_scalar_size; ++j)
      {
        out_buffer[output_vectorized_size + j] = out_buf[j];
      }
    }
  }

  // Iterate over 'in_depth', reduce over 'depth_multiplier', write 'output'.
  if (depth_multiplier > 1)
  {
    for (int64_t d = 0; d < in_depth; ++d)
    {
      const int64_t index = d * depth_multiplier;
      T accum = static_cast<T>(0);
      for (int64_t dm = 0; dm < dm_vectorized_size; dm += kPacketSize)
      {
        const auto v = Eigen::internal::ploadu<Packet>(out_buffer + index + dm);
        accum += Eigen::internal::predux(v);
      }
      // Copy scalar portion of replicated output.
      for (int64_t dm = 0; dm < dm_scalar_size; ++dm)
      {
        accum += out_buffer[index + dm_vectorized_size + dm];
      }
      // Copy to output.
      output[base_output_index + d] = accum;
    }
  }
}

// Computes the depthwise conv2d backprop input of 'out_backprop' by
// 'depthwise_filter' and stores the result in 'in_backprop'.
template <typename T> struct LaunchDepthwiseConvBackpropInputOp<CPUDevice, T>
{
  typedef typename Eigen::internal::packet_traits<T>::type Packet;

  void operator()(int batch, int in_rows, int in_cols, int in_depth, int filter_rows,
                  int filter_cols, int depth_multiplier, int stride, int pad_rows, int pad_cols,
                  int out_rows, int out_cols, int out_depth, const T *out_backprop,
                  const T *depthwise_filter, T *padded_filter_data, T *in_backprop, bool pad_filter,
                  T *out_bprop, T *in_bprop)
  {
    const Eigen::ThreadPoolDevice &d = *eigen_support::GetThreadPoolDevice();

    // Pad 'depthwise_filter' to vector register width (if needed).
    if (pad_filter)
    {
      // Write out padded filter.
      functor::DepthwiseFilterPadOp<T>()(
        batch, in_rows, in_cols, in_depth, filter_rows, filter_cols, depth_multiplier, stride,
        pad_rows, pad_cols, out_rows, out_cols, out_depth, depthwise_filter, padded_filter_data);
    }
    const T *filter_data = pad_filter ? padded_filter_data : depthwise_filter;

    // Computes one shard of depthwise conv2d backprop input.
    auto shard = [d, in_rows, in_cols, in_depth, out_rows, out_cols, out_depth, batch, filter_rows,
                  filter_cols, depth_multiplier, stride, pad_rows, pad_cols, out_backprop,
                  filter_data, in_backprop, out_bprop, in_bprop](int64_t start, int64_t limit) {
      static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));

      const int64_t input_image_size = in_rows * in_cols * in_depth;
      const int64_t output_image_size = out_rows * out_cols * out_depth;
      const int64_t filter_spatial_size = filter_rows * filter_cols;
      const int64_t padded_filter_inner_dim_size =
        ((out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;
      const int64_t out_bprop_size = filter_spatial_size * padded_filter_inner_dim_size;

      int cur_id = d.currentThreadId() + 1;
      assert(cur_id >= 0 && cur_id < d.numThreads() + 1);

      // Use out_bprop buffer to copy regions from 'out_backprop'.
      T *out_bprop_buf = out_bprop + cur_id * out_bprop_size;

      // Use in_bprop buffer for intermediate results.
      T *in_bprop_buf = in_bprop + cur_id * padded_filter_inner_dim_size;

      for (int64_t b = start; b < limit; ++b)
      {
        for (int64_t in_r = 0; in_r < in_rows; ++in_r)
        {
          for (int64_t in_c = 0; in_c < in_cols; ++in_c)
          {
            // Populate 'out_bprop_buf' from local 'out_backprop' region.
            CopyOutputBackpropRegion<T>(batch, in_rows, in_cols, in_depth, filter_rows, filter_cols,
                                        depth_multiplier, stride, pad_rows, pad_cols, out_rows,
                                        out_cols, out_depth, padded_filter_inner_dim_size, in_r,
                                        in_c, out_backprop + b * output_image_size, out_bprop_buf);

            // Compute depthwise backprop input.
            ComputeBackpropInput<T>(
              batch, in_rows, in_cols, in_depth, filter_rows, filter_cols, depth_multiplier, stride,
              pad_rows, pad_cols, out_rows, out_cols, out_depth, padded_filter_inner_dim_size, in_r,
              in_c, filter_data, out_bprop_buf, in_bprop_buf, in_backprop + b * input_image_size);
          }
        }
      }
    };

    const int64_t input_bytes = out_rows * out_cols * out_depth * sizeof(T);
    const int64_t output_bytes = in_rows * in_cols * in_depth * sizeof(T);
    const int64_t compute_cycles = in_rows * in_cols * out_depth * batch;
    const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
    d.parallelFor(batch, cost, shard);
  }
};

template <typename T>
static void
DepthwiseConvBackpropInputReference(int batch, int in_rows, int in_cols, int in_depth, int out_rows,
                                    int out_cols, int out_depth, int stride, int depth_multiplier,
                                    int filter_rows, int filter_cols, int pad_rows, int pad_cols,
                                    const T *out_backprop, const T *filter, T *in_backprop)
{
  // Naive for loop as a reference point without concerns about performance.
  for (int b = 0; b < batch; ++b)
  {
    for (int in_r = 0; in_r < in_rows; ++in_r)
    {
      for (int in_c = 0; in_c < in_cols; ++in_c)
      {
        for (int in_d = 0; in_d < in_depth; ++in_d)
        {
          T sum = 0;
          const int out_d_start = in_d * depth_multiplier;
          const int out_d_end = out_d_start + depth_multiplier;

          for (int out_d = out_d_start; out_d < out_d_end; ++out_d)
          {
            const int out_r_start = std::max(0, (in_r - filter_rows + pad_rows + stride) / stride);
            const int out_r_end = std::min(out_rows - 1, (in_r + pad_rows) / stride);

            for (int out_r = out_r_start; out_r <= out_r_end; ++out_r)
            {
              const int out_c_start =
                std::max(0, (in_c - filter_cols + pad_cols + stride) / stride);
              const int out_c_end = std::min(out_cols - 1, (in_c + pad_cols) / stride);

              for (int out_c = out_c_start; out_c <= out_c_end; ++out_c)
              {
                int f_r = in_r + pad_rows - out_r * stride;
                int f_c = in_c + pad_cols - out_c * stride;
                int filter_dm = out_d - out_d_start;
                int out_backprop_offset =
                  out_d + out_depth * (out_c + out_cols * (out_r + out_rows * b));
                int filter_offset =
                  filter_dm + depth_multiplier * (in_d + in_depth * (f_c + filter_cols * f_r));
                sum += out_backprop[out_backprop_offset] * filter[filter_offset];
              }
            }
          }

          int in_backprop_offset = in_d + in_depth * (in_c + in_cols * (in_r + in_rows * b));
          in_backprop[in_backprop_offset] = sum;
        }
      }
    }
  }
}

// Kernels to compute the gradients of the filters for depthwise convolution.

// Computes filter backprop using 'out_backprop' and 'input_buffer', storing the
// result in 'output_buffer' at an index computed from 'out_r' and 'out_c'.
//
// EX:
//   in_depth = 3, depth_multiplier = 2, filter [2, 2], register_width = 4
//   Both 'input_buffer' and 'filter' are padded to register-width boundaries.
//
//   'input_buffer' [rows, cols, in_depth, depth_multiplier]
//
//     [f00, f01, f10, f11] [f20, f21, 0, 0]   in_row = 0, in_col = 0
//     [e00, e01, e10, e11] [e20, e21, 0, 0]   in_row = 0, in_col = 1
//     [b00, b01, b10, b11] [b20, b21, 0, 0]   in_row = 1, in_col = 0
//     [a00, a01, a10, a11] [a20, a21, 0, 0]   in_row = 1, in_col = 1
//
//   'out_backprop' [out_rows, out_cols, in_depth, depth_multiplier]
//
//     [q00, q01, q10, q11] [q20, q21, r00, r01]
//     [r10, r11, r20, r21] [s00, s01, s10, s11]
//     [s20, s21, t00, t01] [t10, t11, t20, a21]
//
//   First output register of 'filter_backprop'
//     [u0, v0, w0, x0] += ([f00, f01, f10, f11] x [q00, q01, q10, q11])
//
template <typename T>
static void ComputeBackpropFilter(int, int, int, int, int filter_rows, int filter_cols, int, int,
                                  int, int, int out_rows, int out_cols, int out_depth_,
                                  const int64_t padded_out_depth_size, const int64_t out_r,
                                  const int64_t out_c, const T *out_backprop, const T *input_buffer,
                                  T *output_buffer)
{
  typedef typename Eigen::internal::packet_traits<T>::type Packet;
  static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));
  // Calculate vectorized size of 'padded_out_depth_size'.
  const int64_t out_depth = out_depth_;
  const int64_t filter_spatial_size = filter_rows * filter_cols;
  const int64_t output_vectorized_size = (padded_out_depth_size / kPacketSize) * kPacketSize;
  const int64_t base_output_index = (out_r * out_cols + out_c) * out_depth;
  // Determine whether we can execute fast or slow code path.
  const int64_t output_image_size = out_rows * out_cols * out_depth;
  const int64_t output_last_vector_index =
    output_image_size - (filter_spatial_size * padded_out_depth_size);
  const bool fast_path = base_output_index <= output_last_vector_index;

  if (fast_path)
  {
    // TODO(andydavis) Process multiple inputs in 'input_buffer' so we can
    // amortize the cost of 'output_buffer' load store in the loop below.
    for (int i = 0; i < output_vectorized_size; i += kPacketSize)
    {
      // Load vector register from 'out_backprop'.
      const auto out_bprop_block =
        Eigen::internal::ploadu<Packet>(out_backprop + base_output_index + i);
      for (int j = 0; j < filter_spatial_size; ++j)
      {
        const int64_t index = i + j * padded_out_depth_size;
        // Load vector register from 'input_buffer'.
        const auto input_block = Eigen::internal::ploadu<Packet>(input_buffer + index);
        // Load output block into vector register.
        auto out_block_data = output_buffer + index;
        auto out_block = Eigen::internal::ploadu<Packet>(out_block_data);
        // Vector multiply-add.
        out_block = Eigen::internal::pmadd<Packet>(out_bprop_block, input_block, out_block);
        // Store 'out_block' back to memory.
        Eigen::internal::pstoreu<T>(out_block_data, out_block);
      }
    }
  }
  else
  {
    // Slow path (cant do vector reads from non-padded 'out_backprop'.
    for (int i = 0; i < output_vectorized_size; i += kPacketSize)
    {
      // Calculate safe read size from 'out_backprop'.
      const int64_t out_bprop_index = base_output_index + i;
      const int64_t out_bprop_limit = std::min(output_image_size, out_bprop_index + kPacketSize);
      T out_buf[kPacketSize];
      memset(&out_buf, 0, kPacketSize * sizeof(T));
      const int64_t scalar_size = out_bprop_limit - out_bprop_index;
      for (int64_t j = 0; j < scalar_size; ++j)
      {
        out_buf[j] = out_backprop[out_bprop_index + j];
      }
      // Load vector register from 'out_buf'.
      const auto out_bprop_block = Eigen::internal::ploadu<Packet>(out_buf);
      for (int j = 0; j < filter_spatial_size; ++j)
      {
        const int64_t index = i + j * padded_out_depth_size;
        // Load vector register from 'input_buffer'.
        const auto input_block = Eigen::internal::ploadu<Packet>(input_buffer + index);
        // Load output block into vector register.
        auto out_block_data = output_buffer + index;
        auto out_block = Eigen::internal::ploadu<Packet>(out_block_data);
        // Vector multiply-add.
        out_block = Eigen::internal::pmadd<Packet>(out_bprop_block, input_block, out_block);
        // Store 'out_block' back to memory.
        Eigen::internal::pstoreu<T>(out_block_data, out_block);
      }
    }
  }
}

template <typename T> struct LaunchDepthwiseConvBackpropFilterOp<CPUDevice, T>
{
  typedef typename Eigen::internal::packet_traits<T>::type Packet;

  void operator()(int batch, int in_rows, int in_cols, int in_depth, int filter_rows,
                  int filter_cols, int depth_multiplier, int stride, int pad_rows, int pad_cols,
                  int out_rows, int out_cols, int out_depth, const T *out_backprop, const T *input,
                  T *filter_backprop, T *padded_filter_data, T *in_bprop)
  {
    const Eigen::ThreadPoolDevice &d = *eigen_support::GetThreadPoolDevice();

    static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));

    const int64_t filter_spatial_size = filter_rows * filter_cols;
    const int64_t padded_out_depth_size =
      ((out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

    T *output_buffer_data = padded_filter_data;

    // Computes one shard of depthwise conv2d backprop filter.
    // auto shard = [&ctx, &args, &out_backprop, &input, &output_buffer_data](
    auto shard = [&](int64_t start, int64_t limit) {
      static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));
      const int64_t filter_spatial_size = filter_rows * filter_cols;
      const int64_t padded_out_depth_size =
        ((out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

      int cur_id = d.currentThreadId() + 1;
      assert(cur_id >= 0 && cur_id < d.numThreads() + 1);

      const int64_t input_image_size = in_rows * in_cols * in_depth;
      const int64_t output_image_size = out_rows * out_cols * out_depth;
      const int64_t padded_filter_size = filter_spatial_size * padded_out_depth_size;

      T *input_buffer_data = in_bprop + cur_id * padded_filter_size;

      for (int b = start; b < limit; ++b)
      {
        // Initialize 'output_buffer' for 'b'.
        auto *output_buffer = output_buffer_data + b * padded_filter_size;
        memset(output_buffer, 0, padded_filter_size * sizeof(T));

        for (int out_r = 0; out_r < out_rows; ++out_r)
        {
          for (int out_c = 0; out_c < out_cols; ++out_c)
          {
            // Populate 'input_buffer_data' with data from local input region.
            functor::DepthwiseInputCopyOp<T>()(
              batch, in_rows, in_cols, in_depth, filter_rows, filter_cols, depth_multiplier, stride,
              pad_rows, pad_cols, out_rows, out_cols, out_depth, padded_out_depth_size, out_r,
              out_c, input + b * input_image_size, input_buffer_data);
            // Compute depthwise backprop filter.
            ComputeBackpropFilter(
              batch, in_rows, in_cols, in_depth, filter_rows, filter_cols, depth_multiplier, stride,
              pad_rows, pad_cols, out_rows, out_cols, out_depth, padded_out_depth_size, out_r,
              out_c, out_backprop + b * output_image_size, input_buffer_data, output_buffer);
          }
        }
      }
    };

    const int64_t input_bytes = in_rows * in_cols * in_depth * sizeof(T);
    const int64_t output_bytes = out_rows * out_cols * out_depth * sizeof(T);
    const int64_t compute_cycles = out_rows * out_cols * out_depth * batch;
    const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);
    d.parallelFor(batch, cost, shard);

    // Accumulate 'output_buffer' from each shard into 'output'.
    // const int64_t out_depth = out_depth;
    const int64_t vectorized_size = (out_depth / kPacketSize) * kPacketSize;
    const int64_t scalar_size = out_depth - vectorized_size;
    const int64_t padded_filter_size = filter_spatial_size * padded_out_depth_size;
    memset(filter_backprop, 0, filter_spatial_size * out_depth * sizeof(T));

    for (int64_t i = 0; i < filter_spatial_size; ++i)
    {
      const int64_t buffer_base = i * padded_out_depth_size;
      const int64_t output_base = i * out_depth;
      // Write vectorized length of filter's inner dimension to output.
      for (int64_t j = 0; j < vectorized_size; j += kPacketSize)
      {
        // Load data from 'filter_backprop' into vector register.
        auto out_block_data = filter_backprop + output_base + j;
        auto out_block = Eigen::internal::ploadu<Packet>(out_block_data);
        for (int b = 0; b < batch; ++b)
        {
          // Load data from 'output_buffer' for 'b'.
          const auto *output_buffer = output_buffer_data + b * padded_filter_size;
          const auto v = Eigen::internal::ploadu<Packet>(output_buffer + buffer_base + j);
          // Add 'v' to 'out_block'.
          out_block = Eigen::internal::padd<Packet>(out_block, v);
        }
        // Store 'out_block' back to memory.
        Eigen::internal::pstoreu<T>(out_block_data, out_block);
      }
      // Write scalar length of filter's inner dimension to output.
      for (int64_t j = 0; j < scalar_size; ++j)
      {
        for (int b = 0; b < batch; ++b)
        {
          const auto *output_buffer = output_buffer_data + b * padded_filter_size;
          filter_backprop[output_base + vectorized_size + j] +=
            output_buffer[buffer_base + vectorized_size + j];
        }
      }
    }
  }
};

template <typename T>
static void DepthwiseConvBackpropFilterReference(int batch, int in_rows, int in_cols, int in_depth,
                                                 int out_rows, int out_cols, int out_depth,
                                                 int stride, int depth_multiplier, int filter_rows,
                                                 int filter_cols, int pad_rows, int pad_cols,
                                                 const T *out_backprop, const T *input,
                                                 T *filter_backprop)
{
  int num_filter_backprop = filter_rows * filter_cols * in_depth * depth_multiplier;
  memset(filter_backprop, 0, num_filter_backprop * sizeof(T));
  // Naive for loop as a reference point without concerns about performance.
  for (int b = 0; b < batch; ++b)
  {
    for (int out_r = 0; out_r < out_rows; ++out_r)
    {
      for (int out_c = 0; out_c < out_cols; ++out_c)
      {
        for (int out_d = 0; out_d < out_depth; ++out_d)
        {
          const int in_d = out_d / depth_multiplier;
          const int dm = out_d % depth_multiplier;
          const int in_r_start = out_r * stride - pad_rows;
          const int in_c_start = out_c * stride - pad_cols;

          for (int f_r = 0; f_r < filter_rows; ++f_r)
          {
            for (int f_c = 0; f_c < filter_cols; ++f_c)
            {
              const int in_r = in_r_start + f_r;
              const int in_c = in_c_start + f_c;

              if (in_r >= 0 && in_r < in_rows && in_c >= 0 && in_c < in_cols)
              {
                int out_backprop_offset =
                  out_d + out_depth * (out_c + out_cols * (out_r + out_rows * b));
                int input_offset = in_d + in_depth * (in_c + in_cols * (in_r + in_rows * b));
                int filter_backprop_offset =
                  dm + depth_multiplier * (in_d + in_depth * (f_c + filter_cols * f_r));
                filter_backprop[filter_backprop_offset] +=
                  input[input_offset] * out_backprop[out_backprop_offset];
              }
            }
          }
        }
      }
    }
  }
}

} // namespace depthwise_conv_op
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_EIGEN_DEPTHWISE_CONV_OP_H__
