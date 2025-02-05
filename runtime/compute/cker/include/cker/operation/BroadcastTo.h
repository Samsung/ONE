/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_BROADCAST_TO_H__
#define __NNFW_CKER_BROADCAST_TO_H__

#include "cker/Types.h"
#include "cker/Shape.h"
#include "cker/Utils.h"

#include "cker/eigen/EigenSupport.h"

#include "cker/operation/Helper/Tensor.h"
#include "cker/operation/Helper/BCast.h"

#include <vector>

#define UNUSED(x) (void)(x)

namespace nnfw
{
namespace cker
{
namespace functor
{
static const int32_t kint32max = ((int32_t)0x7FFFFFFF);

template <typename Device, typename T> struct FillFunctor
{
  // Computes on device "d": out = out.constant(in(0)),
  void operator()(const Device &d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in);
};

template <typename T> struct FillFunctor<Eigen::ThreadPoolDevice, T>
{
  void operator()(const Eigen::ThreadPoolDevice &d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstScalar in)
  {
    out.device(d) = out.constant(in());
  }
};

template <typename Device, typename T> struct BroadcastTo
{
  template <int NDIMS>
  void DoBCast32Bit(const Device &device, typename TTypes<T, NDIMS>::Tensor out,
                    typename TTypes<T, NDIMS>::ConstTensor in,
                    const typename Eigen::array<int, NDIMS> &bcast) const
  {
    To32Bit(out).device(device) = To32Bit(in).broadcast(bcast);
  }

  template <int NDIMS>
  void DoBCast(const Device &device, typename TTypes<T, NDIMS>::Tensor out,
               typename TTypes<T, NDIMS>::ConstTensor in,
               const typename Eigen::array<Eigen::DenseIndex, NDIMS> &bcast) const
  {
    out.device(device) = in.broadcast(bcast);
  }

  template <int NDIMS>
  void ReshapeAndBCast(const Device &device, Tensor &output_tensor, const Tensor &input_tensor,
                       const BCast &bcast) const
  {
    const bool can_use_32bit = std::is_same<Eigen::GpuDevice, Device>::value &&
                               output_tensor.shape.FlatSize() < kint32max &&
                               input_tensor.shape.FlatSize() < kint32max;
    if (can_use_32bit)
    {
      DoBCast32Bit<NDIMS>(device, output_tensor.template shaped<T, NDIMS>(bcast.result_shape()),
                          input_tensor.template shaped<T, NDIMS>(bcast.x_reshape()),
                          BCast::ToIndexArrayType<int, NDIMS>(bcast.x_bcast()));
    }
    else
    {
      DoBCast<NDIMS>(device, output_tensor.template shaped<T, NDIMS>(bcast.result_shape()),
                     input_tensor.template shaped<T, NDIMS>(bcast.x_reshape()),
                     BCast::ToIndexArrayType<Eigen::DenseIndex, NDIMS>(bcast.x_bcast()));
    }
  }

  // PRECONDITION: rank(input_shape) > 0 &&
  //               rank(input_shape) <= rank(output_shape)  &&
  //               output_shape.num_elements() > 0.
  void operator()(const Device &device, Tensor &output_tensor, const Shape &output_shape,
                  const Tensor &input_tensor, const Shape &input_shape, const BCast &bcast) const
  {
    const int ndims = bcast.y_reshape().size();
    switch (ndims)
    {
      case 1:
        ReshapeAndBCast<1>(device, output_tensor, input_tensor, bcast);
        break;
      case 2:
        ReshapeAndBCast<2>(device, output_tensor, input_tensor, bcast);
        break;
      case 3:
        ReshapeAndBCast<3>(device, output_tensor, input_tensor, bcast);
        break;
      case 4:
        ReshapeAndBCast<4>(device, output_tensor, input_tensor, bcast);
        break;
      case 5:
        ReshapeAndBCast<5>(device, output_tensor, input_tensor, bcast);
        break;
      default:
        // NOTE : UNUSED leaves for maintenance purposes.
        UNUSED(output_shape);
        UNUSED(input_shape);
        break;
    }
  }
};
} // namespace functor

template <typename T>
inline void BroadcastTo(const Shape &input_shape, T *input_data, const Shape &output_shape,
                        T *output_data)
{
  const int input_flatsize = input_shape.FlatSize();

  if (input_shape == output_shape)
  {
    memcpy(output_data, input_data, input_flatsize * sizeof(T));
    return;
  }

  // Input shape's rank must be no greater than rank of output shape.
  assert(input_shape.DimensionsCount() <= output_shape.DimensionsCount());

  // It shouldn't be 0.
  assert(output_shape.DimensionsCount());

  Tensor output_tensor;
  Tensor input_tensor;

  input_tensor.shape.ReplaceWith(input_shape.DimensionsCount(), input_shape.DimsData());
  input_tensor.buffer = input_data;

  output_tensor.shape.ReplaceWith(output_shape.DimensionsCount(), output_shape.DimsData());
  output_tensor.buffer = output_data;

  const Eigen::ThreadPoolDevice &device = *eigen_support::GetThreadPoolDevice();

  // Handle broadcast from Scalar.
  if (input_flatsize == 0)
  {
    functor::FillFunctor<Eigen::ThreadPoolDevice, T>()(device, output_tensor.flat<T>(),
                                                       input_tensor.scalar<T>());
  }

  BCast bcast(BCast::FromShape(input_shape), BCast::FromShape(output_shape),
              /*fewer_dims_optimization=*/true);

  // Predict TRUE.
  assert(bcast.IsValid());
  // should be same.
  assert(BCast::ToShape(bcast.output_shape()) == output_shape);

  functor::BroadcastTo<Eigen::ThreadPoolDevice, T>()(device, output_tensor, output_shape,
                                                     input_tensor, input_shape, bcast);
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_BROADCAST_TO_H__
