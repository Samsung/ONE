/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_EIGEN_XENT_OPS_H__
#define __NNFW_CKER_EIGEN_XENT_OPS_H__

// From tensorflow/core/kernels/xent_op.cc
#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"
#include "cker/operation/Helper/Tensor.h"

// From tensorflow/core/kernels/xent_op.h
namespace nnfw
{
namespace cker
{
namespace xent_ops
{
namespace functor
{

// Functor used by XentOp to do the computations.
template <typename Device, typename T> struct XentFunctor
{
  // Computes Cross Entropy loss and backprop.
  //
  // logits: batch_size, num_classes.
  // labels: batch_size, num_classes.
  // scratch: temporary tensor, dims: batch_size, 1
  // loss: output tensor for the loss, dims: batch_size.
  // backprop: output tensor for the backprop, dims: batch_size, num_classes.
  void operator()(const Device &d, const Eigen::DSizes<Eigen::DenseIndex, 2> &shape,
                  const Eigen::array<Eigen::DenseIndex, 2> &logits_bcast,
                  const Eigen::array<Eigen::DenseIndex, 2> &labels_bcast,
                  typename TTypes<T>::ConstMatrix logits, typename TTypes<T>::ConstMatrix labels,
                  typename TTypes<T>::Matrix scratch, typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop, T reduction_size);
};

} // namespace functor
} // namespace xent_ops
} // namespace cker
} // namespace nnfw

// From tensorflow/core/kernels/xent_op.cc
namespace nnfw
{
namespace cker
{
namespace xent_ops
{

// Enable CPUDevice only for xent_ops
using CPUDevice = Eigen::ThreadPoolDevice;
using Index = Eigen::Index;

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor
{
template <typename Device, typename T> struct XentFunctorBase
{
  void operator()(const Device &d, const Eigen::DSizes<Eigen::DenseIndex, 2> &shape,
                  const Eigen::array<Eigen::DenseIndex, 2> &logits_bcast,
                  const Eigen::array<Eigen::DenseIndex, 2> &labels_bcast,
                  typename TTypes<T>::ConstMatrix logits, typename TTypes<T>::ConstMatrix labels,
                  typename TTypes<T>::Matrix scratch, typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop, T reduction_size)
  {
    T *scratch_ptr = scratch.data();
    T *backprop_ptr = backprop.data();

    T *loss_ptr = loss.data();

    int row_size = shape[1];

    if (shape[0] > 0)
    {
      backprop.device(d) = logits.broadcast(logits_bcast);
      scratch.device(d) = labels.broadcast(labels_bcast);
      auto reductionWorker = [&](int64_t begin, int64_t end) -> void {
        for (int i = begin; i < end; i++)
        {
          T *this_backprop = backprop_ptr + (i * row_size);
          T *this_logits = backprop_ptr + (i * row_size);
          T *this_labels = scratch_ptr + (i * row_size);
          T max_logits = this_logits[0];

          // calculating max_logits
          for (int j = 1; j < row_size; j++)
          {
            max_logits = std::max(max_logits, this_logits[j]);
          }

          T sum = T(0);
          T loss_sum = T(0);

          for (int j = 0; j < row_size; j++)
          {
            // Note that if input is reused than this_logits and this_backprop
            // is same buffer, so after this calculation this_logits should no
            // longer be trusted
            this_backprop[j] = this_logits[j] - max_logits;
            sum = sum + exp(this_backprop[j]);
          }

          // loss calculation
          T log_sum = log(sum);
          for (int j = 0; j < row_size; j++)
          {
            loss_sum += this_labels[j] * (log_sum - this_backprop[j]);
            this_backprop[j] = ((exp(this_backprop[j]) / sum) - this_labels[j]) / reduction_size;
          }
          loss_ptr[i] = loss_sum;
        }
      };
      const int64_t compute_cycles = 50 * row_size;
      const int64_t input_bytes = sizeof(T) * row_size;
      const int64_t output_bytes = sizeof(T) * row_size;
      const Eigen::TensorOpCost cost(input_bytes, output_bytes, compute_cycles);

      d.parallelFor(shape[0], cost, reductionWorker);
    }
  }
};

template <typename T> struct XentFunctor<CPUDevice, T> : XentFunctorBase<CPUDevice, T>
{
};

} // namespace functor
} // namespace xent_ops
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_EIGEN_XENT_OPS_H__
