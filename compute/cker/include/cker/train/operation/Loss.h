/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_TRAIN_OPERATION_LOSS_H__
#define __NNFW_CKER_TRAIN_OPERATION_LOSS_H__

#include <numeric>

#include "cker/Shape.h"
#include "cker/eigen/Utils.h"
#include "cker/eigen/xent_op.h"
#include "cker/operation/Helper/BCast.h"

namespace nnfw
{
namespace cker
{
namespace train
{

template <typename T>
inline void MSE(const Shape &y_pred_shape, const T *y_pred_data, const Shape &y_true_shape,
                const T *y_true_data, const Shape &, T *output_data)
{
  // TODO Consider Reduction
  // if (output_shape != Shape{1})
  //   throw std::runtime_error("cker::MSE: output_shape != Shape{1}");
  if (y_pred_shape != y_true_shape)
    throw std::runtime_error("cker::MSE: y_pred_shape != y_true_shape");

  const auto y_pred = MapAsMatrixWithLastDimAsRows(y_pred_data, y_pred_shape);
  const auto y_true = MapAsMatrixWithLastDimAsRows(y_true_data, y_true_shape);

  double squared_sum;
  for (size_t c = 0; c < (size_t)y_pred.cols(); ++c)
  {
    squared_sum = 0.0f;
    for (size_t r = 0; r < (size_t)y_pred.rows(); ++r)
    {
      double error = y_pred.coeff(r, c) - y_true.coeff(r, c);
      squared_sum += (error * error);
    }
    output_data[c] = static_cast<T>(squared_sum / y_pred.rows());
  }
}

template <typename T>
inline void MSEGrad(const Shape &y_pred_shape, const T *y_pred_data, const Shape &y_true_shape,
                    const T *y_true_data, const Shape &grad_shape, T *grad_data)
{
  if (y_pred_shape != y_true_shape)
    throw std::runtime_error("cker::MSEGrad: y_pred_shape != y_true_shape");
  if (y_pred_shape != grad_shape)
    throw std::runtime_error("cker::MSEGrad: y_pred_shape != grad_shape");

  const int size = grad_shape.FlatSize();
  for (int i = 0; i < size; ++i)
  {
    grad_data[i] = static_cast<T>(-2 * (y_true_data[i] - y_pred_data[i]) / size);
  }
}

template <typename T> bool checkValue(const T *data, int size, T min, T max)
{
  for (int i = 0; i < size; ++i)
  {
    if (data[i] > max || data[i] < min)
      return false;
  }
  return true;
}

template <typename T>
inline void CategoricalCrossEntropy(const Shape &y_pred_shape, const T *y_pred_data,
                                    const Shape &y_true_shape, const T *y_true_data,
                                    const Shape &output_shape, T *output_data,
                                    const Shape &backprop_shape, T *backprop_data)
{
  Tensor logits;
  Tensor labels;
  Tensor scratch;
  Tensor loss_out;
  Tensor back_out;

  logits.shape.ReplaceWith(y_pred_shape);
  logits.buffer = const_cast<T*>(y_pred_data);
  const Tensor& logits_in = logits;

  labels.shape.ReplaceWith(y_true_shape);
  labels.buffer = const_cast<T*>(y_true_data);
  const Tensor& labels_in = labels;

  Shape shape_in(logits_in.shape);

  // TODO Support Broadcasting
  nnfw::cker::BCast bcast(nnfw::cker::BCast::FromShape(y_pred_shape),
                          nnfw::cker::BCast::FromShape(y_true_shape),
                          /*fewer_dims_optimization=*/false);
  // if (y_pred_size != y_true_size)
  // {
  //   if (!bcast.IsValid())
  //   {
  //     throw std::runtime_error("cker::CategoricalCrossEntropy: logits and labels must be broadcastable: logits_size=" +
  //                               std::to_string(y_pred_shape.FlatSize()) + " label_size=" + std::to_string(y_true_shape.FlatSize()));
  //   }
  //   shape_in = nnfw::cker::BCast::ToShape(bcast.output_shape());
  // }

  Shape scratch_shape({shape_in.Dims(0), 1});
  std::vector<T> scratch_vec(scratch_shape.FlatSize());
  scratch.shape.ReplaceWith(scratch_shape);
  scratch.buffer = scratch_vec.data();

  loss_out.shape.ReplaceWith(output_shape);
  loss_out.buffer = output_data;

  back_out.shape.ReplaceWith(backprop_shape);
  back_out.buffer = backprop_data;

  const auto shape_in_batches = shape_in.Dims(0);
  const auto shape_in_size = FlatSizeSkipDim(shape_in, 0);
  if (shape_in.Dims(0) > 0)
  {
    const xent_op::CPUDevice &device = *eigen_support::GetThreadPoolDevice();
    xent_op::functor::XentFunctor<xent_op::CPUDevice, T> functor;
    functor(device, Eigen::DSizes<Eigen::DenseIndex, 2>(shape_in_batches, shape_in_size),
          nnfw::cker::BCast::ToIndexArray<2>(bcast.x_bcast()),
          nnfw::cker::BCast::ToIndexArray<2>(bcast.y_bcast()),
          logits_in.template shaped<T, 2>(bcast.x_reshape()),
          labels_in.template shaped<T, 2>(bcast.y_reshape()),
          scratch.matrix<T>(), loss_out.vec<T>(), back_out.matrix<T>());
  }
}

template <typename T>
inline void CategoricalCrossEntropy(const T *y_pred_data, const T *y_true_data, T *output_data,
                                    const int batch_size, const int input_size)
{
  const T *y_prob_data = y_pred_data;

  if (!checkValue(y_pred_data, input_size * batch_size, static_cast<T>(0), static_cast<T>(1)))
  {
    throw std::runtime_error("cker::CategoricalCrossEntropy: y_pred data is not logit data.");
  }

  float sum;
  for (int b = 0; b < batch_size; ++b)
  {
    sum = 0.f;
    int b_offset = b * input_size;
    for (int i = 0; i < input_size; ++i)
    {
      if (y_true_data[b_offset + i] != 0)
      {
        sum += -std::log(std::max(y_prob_data[b_offset + i], static_cast<float>(1.0e-20))) *
                  y_true_data[b_offset + i];
      }
    }
    output_data[b] = sum;
  }
}

template <typename T>
inline void CategoricalCrossEntropyGrad(const T *y_pred_data, const T *y_true_data, T *grad_data,
                                        const int batch_size, const int input_size)
{
  if (!checkValue(y_pred_data, input_size * batch_size, static_cast<T>(0), static_cast<T>(1)))
  {
    throw std::runtime_error("cker::CategoricalCrossEntropyGrad: y_pred data is not logit data.");
  }

  for (int b = 0; b < batch_size; ++b)
  {
    int b_offset = b * input_size;
    for (int i = 0; i < input_size; ++i)
    {
      grad_data[b_offset + i] = -(y_true_data[b_offset + i] /
                                  std::max(y_pred_data[b_offset + i], static_cast<float>(1e-20)));
    }
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_LOSS_H__
