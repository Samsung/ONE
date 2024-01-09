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
                  std::vector<uint8_t *> &out_bprop)
  {
    // static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));

    // // Pad 'depthwise_filter' to vector register width (if needed).
    // const bool pad_filter = (out_depth % kPacketSize) == 0 ? false : true;
    // Tensor padded_filter;
    // // Allocate space for padded filter.
    // const int filter_spatial_size = filter_rows * filter_cols;
    // const int padded_filter_inner_dim_size =
    //   ((out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;
    // Shape padded_filter_shape({filter_spatial_size, padded_filter_inner_dim_size});
    // std::vector<T> padded_filter_vec(padded_filter_shape.FlatSize());
    // padded_filter.shape.ReplaceWith(padded_filter_shape);
    // padded_filter.buffer = padded_filter_vec.data();
    if (pad_filter)
    {
      // Write out padded filter.
      functor::DepthwiseFilterPadOp<T>()(
        batch, in_rows, in_cols, in_depth, filter_rows, filter_cols, depth_multiplier, stride,
        pad_rows, pad_cols, out_rows, out_cols, out_depth, depthwise_filter, padded_filter_data);
    }
    const T *filter_data = pad_filter ? padded_filter_data : depthwise_filter;

    // Computes one shard of depthwise conv2d backprop input.
    auto shard = [&](int64_t start, int64_t limit) {
      static const int64_t kPacketSize = (sizeof(Packet) / sizeof(T));

      const int64_t input_image_size = in_rows * in_cols * in_depth;
      const int64_t output_image_size = out_rows * out_cols * out_depth;
      // const int filter_spatial_size = filter_rows * filter_cols;
      const int padded_filter_inner_dim_size =
        ((out_depth + kPacketSize - 1) / kPacketSize) * kPacketSize;

      // // Allocate buffer to copy regions from 'out_backprop'.
      // Tensor out_bprop_buffer;
      // Shape out_bprop_shape({filter_spatial_size, padded_filter_inner_dim_size});
      // std::vector<T> out_bprop_vec(out_bprop_shape.FlatSize());
      // out_bprop_buffer.shape.ReplaceWith(out_bprop_shape);
      // out_bprop_buffer.buffer = out_bprop_vec.data();
      // T *out_bprop_buf = static_cast<T *>(out_bprop_buffer.buffer);
      T *out_bprop_buf = reinterpret_cast<T *>(out_bprop[start]);

      // Allocate buffer for intermediate results.
      Tensor in_bprop_buffer;
      Shape in_bprop_shape({padded_filter_inner_dim_size});
      std::vector<T> in_bprop_vec(in_bprop_shape.FlatSize());
      in_bprop_buffer.shape.ReplaceWith(in_bprop_shape);
      in_bprop_buffer.buffer = in_bprop_vec.data();
      T *in_bprop_buf = static_cast<T *>(in_bprop_buffer.buffer);

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

    // const int shard_cost = in_rows * in_cols * out_depth;
    // auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    // Shard(worker_threads.num_threads, worker_threads.workers, batch,
    //       shard_cost, shard);

    const Eigen::ThreadPoolDevice &d = *eigen_support::GetThreadPoolDevice();
    int input_bytes = in_rows * in_cols * in_depth;
    int output_bytes = out_rows * out_cols * out_depth;
    int compute_cycles = in_rows * in_cols * out_depth * batch;
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
