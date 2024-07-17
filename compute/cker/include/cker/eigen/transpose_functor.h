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

// #ifndef __NNFW_CKER_EIGEN_TRANSPOSE_FUNCTOR_H__
// #define __NNFW_CKER_EIGEN_TRANSPOSE_FUNCTOR_H__

// // tensorflow/core/kernels/transpose_functor_cpu.cc
// #define EIGEN_USE_THREADS

// #include <thread>
// #include <absl/container/inlined_vector.h>
// #include <absl/types/span.h>
// #include "unsupported/Eigen/CXX11/Tensor"
// #include "cker/operation/Helper/Tensor.h"

// typedef Eigen::ThreadPoolDevice CPUDevice;

// // template <typename T>
// // using ArraySlice ABSL_DEPRECATE_AND_INLINE() = absl::Span<const T>;

// // template <typename T, size_t N>
// // using InlinedVector ABSL_DEPRECATE_AND_INLINE() = absl::InlinedVector<T, N>;

// namespace nnfw
// {
// namespace cker
// {
// namespace transpose_functor
// {

// namespace
// {

// // Helper to compute 'strides' given a tensor 'shape'. I.e.,
// // strides[i] = prod(shape.dim_size[(i+1):])
// template <typename T>
// absl::InlinedVector<T, 8> ComputeStride(const TensorShape& shape) {
//   const int ndims = shape.DimensionsCount();
//   absl::InlinedVector<T, 8> strides(ndims);
//   T stride = 1;
//   for (int i = ndims - 1; i >= 0; --i) {
//     strides[i] = stride;
//     stride *= static_cast<T>(shape.dim_size(i));
//   }
//   return strides;
// }

// template <typename T, bool conjugate>
// void TransposeSimple(const CPUDevice& device, const Tensor& in,
//                      const absl::Span<int32_t> perm, Tensor* out) {
//   const int ndims = in.shape.DimensionsCount();
//   absl::InlinedVector<int64_t, 8> in_strides =
//       ComputeStride<int64_t>(in.shape());
//   absl::InlinedVector<int64_t, 8> out_strides =
//       ComputeStride<int64_t>(out->shape());
//   const T* p = reinterpret_cast<const T*>(in.base());
//   T* q = reinterpret_cast<T*>(const_cast<char*>((out->base())));
//   auto transpose_fn = [=, &in_strides, &out_strides, &perm](int64_t begin,
//                                                             int64_t end) {
//     for (int64_t o_idx = begin; o_idx < end; ++o_idx) {
//       int64_t i_idx = 0;
//       int64_t t = o_idx;
//       for (int i = 0; i < ndims; ++i) {
//         const int64_t ratio = t / out_strides[i];
//         t -= ratio * out_strides[i];
//         i_idx += ratio * in_strides[perm[i]];
//       }
//       if (conjugate) {
//         q[o_idx] = Eigen::numext::conj(p[i_idx]);
//       } else {
//         q[o_idx] = p[i_idx];
//       }
//     }
//   };
//   double cycles_per_element =
//       (conjugate ? 1 : 0) +
//       ndims * (Eigen::TensorOpCost::DivCost<int64_t>() +
//                2 * Eigen::TensorOpCost::MulCost<int64_t>() +
//                2 * Eigen::TensorOpCost::AddCost<int64_t>());
//   Eigen::TensorOpCost cost(/*bytes_loaded=*/sizeof(T),
//                            /*bytes_stored=*/sizeof(T), cycles_per_element);
//   device.parallelFor(in.NumElements(), cost, std::move(transpose_fn));
// }

// } // namespace

// // From tensorflow/core/kernels/transpose_functor.h
// namespace internal
// {

// // Uses Eigen to transpose.
// template <typename Device, typename T, int NDIMS>
// void TransposeUsingEigen(const Device& d, const Tensor& in,
//                          const absl::Span<int32_t> perm, bool conjugate,
//                          Tensor* out) {
//   Eigen::array<int, NDIMS> p;
//   for (int i = 0; i < NDIMS; ++i) p[i] = perm[i];
//   auto x = typename TTypes<T, NDIMS>::ConstTensor(
//       reinterpret_cast<const T*>(in.base()),
//       in.shaped());
//   auto y = typename TTypes<T, NDIMS>::Tensor(
//       reinterpret_cast<T*>(const_cast<char*>(out->base())),
//       out->shaped());
//   if (conjugate) {
//     y.device(d) = x.conjugate().shuffle(p);
//   } else {
//     y.device(d) = x.shuffle(p);
//   }
// }

// template <typename Device>
// Status DoTransposeImpl(const Device& d, const Tensor& in,
//                        const absl::Span<const int32_t> perm, bool conjugate,
//                        Tensor* out) {
//   CHECK_EQ(in.dims(), out->dims());
//   CHECK_EQ(in.dims(), perm.size());
//   CHECK_EQ(in.dtype(), out->dtype());
//   switch (in.dtype()) {
//     case DT_BOOL:
//     case DT_INT8:
//     case DT_QINT8:
//     case DT_QUINT8:
//     case DT_UINT8:
//     case DT_FLOAT8_E5M2:
//     case DT_FLOAT8_E4M3FN:
//       Transpose<Device, uint8>::run(d, in, perm, out);
//       break;

//     case DT_BFLOAT16:
//     case DT_HALF:
//     case DT_INT16:
//     case DT_QINT16:
//     case DT_QUINT16:
//     case DT_UINT16:
//       Transpose<Device, uint16>::run(d, in, perm, out);
//       break;

//     case DT_FLOAT:
//     case DT_INT32:
//     case DT_QINT32:
//     case DT_UINT32:
//       Transpose<Device, uint32_t>::run(d, in, perm, out);
//       break;

//     case DT_DOUBLE:
//     case DT_INT64:
//     case DT_UINT64:
//       Transpose<Device, uint64>::run(d, in, perm, out);
//       break;

//     case DT_COMPLEX64:
//       if (conjugate) {
// #if defined(__ANDROID__) and !defined(__clang__)
//         // Workaround for GCC compiler bug in Android toolchain.
//         return errors::Unimplemented(
//             "Conjugate transpose of complex64 not supported for GCC on "
//             "Android.");
// #else
//         Transpose<Device, complex64, /*conjugate=*/true>::run(d, in, perm, out);
// #endif
//       } else {
//         Transpose<Device, uint64>::run(d, in, perm, out);
//       }
//       break;

//     case DT_COMPLEX128:
//       if (conjugate) {
//         Transpose<Device, complex128, /*conjugate=*/true>::run(d, in, perm,
//                                                                out);
//       } else {
//         Transpose<Device, complex128, /*conjugate=*/false>::run(d, in, perm,
//                                                                 out);
//       }
//       break;

//     case DT_STRING:
//       Transpose<Device, tstring>::run(d, in, perm, out);
//       break;

//     default:
//       return errors::Unimplemented("Unsupported dtype on CPU: ", in.dtype());
//   }
//   return absl::OkStatus();
// }

// } // internal

// template <typename T, bool conjugate>
// struct Transpose<CPUDevice, T, conjugate> {
//   static void run(const CPUDevice& d, const Tensor& in,
//                   const absl::Span<int32_t> perm, Tensor* out) {
//     switch (in.dims()) {
//       case 2:
//         internal::TransposeUsingEigen<CPUDevice, T, 2>(d, in, perm, conjugate,
//                                                        out);
//         break;
//       case 3:
//         internal::TransposeUsingEigen<CPUDevice, T, 3>(d, in, perm, conjugate,
//                                                        out);
//         break;
//       case 4:
//         internal::TransposeUsingEigen<CPUDevice, T, 4>(d, in, perm, conjugate,
//                                                        out);
//         break;
//       case 5:
//         internal::TransposeUsingEigen<CPUDevice, T, 5>(d, in, perm, conjugate,
//                                                        out);
//         break;
//       case 6:
//         internal::TransposeUsingEigen<CPUDevice, T, 6>(d, in, perm, conjugate,
//                                                        out);
//         break;
//       case 7:
//         internal::TransposeUsingEigen<CPUDevice, T, 7>(d, in, perm, conjugate,
//                                                        out);
//         break;
//       case 8:
//         internal::TransposeUsingEigen<CPUDevice, T, 8>(d, in, perm, conjugate,
//                                                        out);
//         break;
//       default:
//         TransposeSimple<T, conjugate>(d, in, perm, out);
//         break;
//     }
//   }
// };

// } // transpose_functor
// } // cker
// } // nnfw

// #endif // __NNFW_CKER_EIGEN_TRANSPOSE_FUNCTOR_H__
