/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __NNFW_CKER_TRAIN_OPERATION_LOSS_H__
#define __NNFW_CKER_TRAIN_OPERATION_LOSS_H__

#include <numeric>

#include "cker/Shape.h"
#include "cker/eigen/EigenSupport.h"
#include "cker/eigen/Utils.h"
#include "cker/eigen/xent_op.h"
#include "cker/operation/Helper/BCast.h"
#include "cker/train/Types.h"

namespace nnfw
{
namespace cker
{
namespace train
{

template <typename T> inline T square(T value) { return value * value; }
template <typename T> inline T log_threshold() { return static_cast<T>(1e-20); }

template <typename T>
inline void MSE(const Shape &y_pred_shape, const T *y_pred_data, const Shape &y_true_shape,
                const T *y_true_data, const Shape &output_shape, T *output_data)
{
  if (output_shape.DimensionsCount() != 1)
    throw std::runtime_error("cker::MSE: output dimension count should be 1");
  if (output_shape.Dims(0) != y_pred_shape.Dims(0))
    throw std::runtime_error("cker::MSE: output and y_pred do not have the same batch");
  if (y_pred_shape != y_true_shape)
    throw std::runtime_error("cker::MSE: y_pred_shape != y_true_shape");

  const auto batch = y_pred_shape.Dims(0);
  const auto size = FlatSizeSkipDim(y_pred_shape, 0);

  for (int b = 0; b < batch; ++b)
  {
    float sum = 0.f;
    for (int i = 0; i < size; ++i)
    {
      sum += square(y_pred_data[b * size + i] - y_true_data[b * size + i]);
    }
    output_data[b] = static_cast<T>(sum / size);
  }
}

template <typename T>
inline void MSEGrad(const Shape &y_pred_shape, const T *y_pred_data, const Shape &y_true_shape,
                    const T *y_true_data, const Shape &grad_shape, T *grad_data,
                    LossReductionType reduction_type)
{
  if (y_pred_shape != y_true_shape)
    throw std::runtime_error("cker::MSEGrad: y_pred_shape != y_true_shape");
  if (y_pred_shape != grad_shape)
    throw std::runtime_error("cker::MSEGrad: y_pred_shape != grad_shape");

  const int batch_size = grad_shape.Dims(0);
  const auto flat_size = FlatSizeSkipDim(grad_shape, 0);
  auto reduction_size = 1;
  switch (reduction_type)
  {
    case LossReductionType::SUM_OVER_BATCH_SIZE:
      reduction_size = batch_size * flat_size;
      break;
    case LossReductionType::SUM:
      reduction_size = flat_size;
      break;
    default:
      throw std::runtime_error("Unsupported reduction type");
  }

  for (int b = 0; b < batch_size; ++b)
  {
    for (int i = 0; i < flat_size; ++i)
    {
      const int offset = b * flat_size + i;
      assert(offset >= 0);
      grad_data[offset] =
        static_cast<T>(-2 * (y_true_data[offset] - y_pred_data[offset]) / reduction_size);
    }
  }
}

template <typename T>
inline void CategoricalCrossEntropy(const Shape &y_pred_shape, const T *y_pred_data,
                                    const Shape &y_true_shape, const T *y_true_data,
                                    const Shape &output_shape, T *output_data)
{
  if (output_shape.DimensionsCount() != 1)
    throw std::runtime_error("cker::CategoricalCrossEntropy: output dimension count should be 1");
  if (y_pred_shape != y_true_shape)
    throw std::runtime_error(
      "cker::CategoricalCrossEntropy: y_pred and y_true do not have the same shape");
  if (output_shape.Dims(0) != y_pred_shape.Dims(0))
    throw std::runtime_error(
      "cker::CategoricalCrossEntropy: output and y_pred do not have the same batch");

  const auto y_pred = MapAsMatrixWithLastDimAsRows(y_pred_data, y_pred_shape);
  const auto y_true = MapAsMatrixWithLastDimAsRows(y_true_data, y_true_shape);
  auto output = MapAsVector(output_data, output_shape);

  output = -(y_true.array() * y_pred.array().cwiseMax(log_threshold<T>()).log()).colwise().sum();
}

template <typename T>
inline void CategoricalCrossEntropyGrad(const Shape &y_pred_shape, const T *y_pred_data,
                                        const Shape &y_true_shape, const T *y_true_data,
                                        const Shape &grad_shape, T *grad_data,
                                        LossReductionType reduction_type)
{
  if (y_pred_shape != y_true_shape)
    throw std::runtime_error(
      "cker::CategoricalCrossEntropyGrad: y_pred and y_true do not have the same shape");
  if (y_pred_shape != grad_shape)
    throw std::runtime_error(
      "cker::CategoricalCrossEntropyGrad: y_pred and grad do not have the same shape");

  const auto y_pred = MapAsMatrixWithLastDimAsRows(y_pred_data, y_pred_shape);
  const auto y_true = MapAsMatrixWithLastDimAsRows(y_true_data, y_true_shape);
  auto grad = MapAsMatrixWithLastDimAsRows(grad_data, grad_shape);

  const int32_t batch_size = grad_shape.Dims(0);
  int32_t reduction_size = 1;
  switch (reduction_type)
  {
    case LossReductionType::SUM_OVER_BATCH_SIZE:
      reduction_size = batch_size;
      break;
    case LossReductionType::SUM:
      reduction_size = 1;
      break;
    default:
      throw std::runtime_error("Unsupported reduction type");
  }
  assert(reduction_size > 0);

  grad = -(y_true.array() / y_pred.array().cwiseMax(log_threshold<T>())) /
         static_cast<T>(reduction_size);
}

template <typename T>
void CategoricalCrossEntropyWithLogits(const Shape &logits_shape, const T *logits_data,
                                       const Shape &y_true_shape, const T *y_true_data,
                                       const Shape &loss_out_shape, T *loss_out_data,
                                       const Shape &grad_shape, T *grad_data,
                                       LossReductionType reduction_type)
{
  // TODO Enable broadcast shapes
  if (loss_out_shape.DimensionsCount() != 1)
    throw std::runtime_error(
      "cker::CategoricalCrossEntropyWithLogits: loss output dimension count should be 1");
  if (logits_shape != y_true_shape)
    throw std::runtime_error(
      "cker::CategoricalCrossEntropyWithLogits: logits and y_true do not have the same shape");
  if (loss_out_shape.Dims(0) != logits_shape.Dims(0))
    throw std::runtime_error(
      "cker::CategoricalCrossEntropyWithLogits: loss_out and logits do not have the same batch");
  if (logits_shape != grad_shape)
    throw std::runtime_error(
      "cker::CategoricalCrossEntropyWithLogits: logits and grad do not have the same shape");

  auto shape_in = logits_shape;

  BCast bcast(BCast::FromShape(shape_in), BCast::FromShape(y_true_shape),
              /*fewer_dims_optimization=*/false);

  // loss is 1-D (one per example), and size is batch_size.

  Tensor logits_in;
  Tensor labels_in;
  Tensor scratch;
  Tensor loss_out;
  Tensor back_out;

  logits_in.shape.ReplaceWith(shape_in.DimensionsCount(), shape_in.DimsData());
  logits_in.buffer = const_cast<T *>(logits_data);

  labels_in.shape.ReplaceWith(y_true_shape.DimensionsCount(), y_true_shape.DimsData());
  labels_in.buffer = const_cast<T *>(y_true_data);

  scratch.shape.ReplaceWith(shape_in.DimensionsCount(), shape_in.DimsData());
  std::vector<T> scratch_vec(shape_in.Dims(0) * shape_in.Dims(1), static_cast<T>(0));
  scratch.buffer = scratch_vec.data();

  Shape shape_loss_out{shape_in.Dims(0)};
  loss_out.shape.ReplaceWith(shape_loss_out.DimensionsCount(), shape_loss_out.DimsData());
  loss_out.buffer = loss_out_data;

  back_out.shape.ReplaceWith(shape_in.DimensionsCount(), shape_in.DimsData());
  back_out.buffer = grad_data;

  if (shape_in.Dims(0) > 0)
  {
    const int32_t batch_size = grad_shape.Dims(0);
    int32_t reduction_size = 1;
    switch (reduction_type)
    {
      case LossReductionType::SUM_OVER_BATCH_SIZE:
        reduction_size = batch_size;
        break;
      case LossReductionType::SUM:
        reduction_size = 1;
        break;
      default:
        throw std::runtime_error("Unsupported reduction type");
    }
    assert(reduction_size > 0);

    const xent_ops::CPUDevice &device = *eigen_support::GetThreadPoolDevice();
    xent_ops::functor::XentFunctor<xent_ops::CPUDevice, T> functor;
    const Eigen::DSizes<Eigen::DenseIndex, 2> shape{shape_in.Dims(0), shape_in.Dims(1)};

    functor(device, shape, BCast::ToIndexArray<2>(bcast.x_bcast()),
            BCast::ToIndexArray<2>(bcast.y_bcast()),
            logits_in.template shaped<const T, 2>(bcast.x_reshape()),
            labels_in.template shaped<const T, 2>(bcast.y_reshape()), scratch.matrix<T>(),
            loss_out.vec<T>(), back_out.matrix<T>(), static_cast<T>(reduction_size));
  }
}

} // namespace train
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_TRAIN_OPERATION_LOSS_H__
