// /*
//  * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
//  * Copyright 2015 The TensorFlow Authors. All Rights Reserved.
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  *      http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

// #ifndef __NNFW_CKER_EIGEN_BIAS_OP_H__
// #define __NNFW_CKER_EIGEN_BIAS_OP_H__

// // From tensorflow/core/kernels/bias_op.cc
// #define EIGEN_USE_THREADS

// #include "unsupported/Eigen/CXX11/Tensor"
// #include "cker/operation/Helper/Tensor.h"

// // From tensorflow/core/kernels/bias_op.h
// namespace nnfw
// {
// namespace cker
// {
// namespace bias_op
// {

// namespace functor
// {

// // Functor used by BiasOp to do the computations.
// template <typename Device, typename T>
// struct Bias {
//   // Add "bias" to "input", repeating "bias".
//   void operator()(const Device& d, typename TTypes<T>::ConstFlat input,
//                   typename TTypes<T>::ConstVec bias,
//                   typename TTypes<T>::Flat output) {
//     const Eigen::Index rest_size = input.size() / bias.dimension(0);
//     Eigen::DSizes<Eigen::Index, 1> bcast(rest_size);
//     MaybeWith32BitIndexing<Device>(
//         [&](auto input32, auto bias32, auto output32, const auto& bcast32) {
//           output32.device(d) = input32 + bias32.broadcast(bcast32);
//         },
//         input, bias, output, bcast);
//   }

//   // NCHW layout, repeating on the first dimension, broadcasting on the last
//   // dimension.
//   void operator()(const Device& d, typename TTypes<T>::ConstMatrix input,
//                   typename TTypes<T>::ConstMatrix bias1,  // shape [C, 1].
//                   typename TTypes<T>::Matrix output) {
//     const Eigen::Index rest_size = input.dimension(0) / bias1.dimension(0);
//     Eigen::DSizes<Eigen::Index, 2> bcast(rest_size, input.dimension(1));
//     MaybeWith32BitIndexing<Device>(
//         [&](auto input32, auto bias32, auto output32, const auto& bcast32) {
//           output32.device(d) = input32 + bias32.broadcast(bcast32);
//         },
//         input, bias1, output, bcast);
//   }
// };

// } // namespace functor
// } // namespace bias_op
// } // namespace cker
// } // namespace nnfw

// // From tensorflow/core/kernels/bias_op.cc
// namespace nnfw
// {
// namespace cker
// {
// namespace bias_op
// {

// template <typename Device, typename T>
// class BiasOp : public BinaryOp<T> {
//  public:
//   explicit BiasOp(OpKernelConstruction* context) : BinaryOp<T>(context) {

//   }

//   void Compute(OpKernelContext* context) override {
//     const Tensor& input = context->input(0);
//     const Tensor& bias = context->input(1);

//     OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input.shape()),
//                 errors::InvalidArgument("Input tensor must be at least 2D: ",
//                                         input.shape()));
//     OP_REQUIRES(context, TensorShapeUtils::IsVector(bias.shape()),
//                 errors::InvalidArgument("Biases must be 1D: ", bias.shape()));

//     // Added by intel_tf to support NCHW on CPU regardless of MKL used or not.
//     int channel_dim;
//     if (data_format_ == FORMAT_NCHW) {
//       channel_dim = 1;  // NCHW always have channel dim in 1 (with 3, 4, 5
//                         // dimensions data).
//     } else {
//       channel_dim = input.shape().dims() - 1;  // End of code by intel_tf.
//     }

//     OP_REQUIRES(context,
//                 bias.shape().dim_size(0) == input.shape().dim_size(channel_dim),
//                 errors::InvalidArgument(
//                     "Must provide as many biases as the last dimension "
//                     "of the input tensor: ",
//                     bias.shape(), " vs. ", input.shape()));

//     Tensor* output = nullptr;
//     OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
//                                 {0}, 0, input.shape(), &output));
//     if (input.NumElements() == 0) return;

//     functor::Bias<Device, T> functor;
//     const Device& d = context->eigen_device<Device>();
//     if (data_format_ == FORMAT_NCHW && input.shape().dims() > 2) {
//       functor(d, input.flat_inner_outer_dims<T, 2>(1),
//               bias.flat_outer_dims<T, 2>(),
//               output->flat_inner_outer_dims<T, 2>(1));
//     } else {
//       functor(d, input.flat<T>(), bias.vec<T>(), output->flat<T>());
//     }
//   }

//  private:
//   TensorFormat data_format_;
// };

// } // namespace bias_op
// } // namespace cker
// } // namespace nnfw
// #endif // __NNFW_CKER_EIGEN_BIAS_OP_H__