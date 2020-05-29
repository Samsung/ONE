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

#ifndef __NNFW_CKER_FILL_H__
#define __NNFW_CKER_FILL_H__

#include "cker/Shape.h"

namespace nnfw
{
namespace cker
{
template <typename T, typename U>
inline void FusedBatchNorm( T input_x, T input_scale,
                            T input_offset, float *epsilon,
                            T estimated_mean_input, T estimated_variance_input,
                            T batch_mean_output, T batch_var_output)
{
    typename TTypes<T, 4>::Tensor x(transformed_x.tensor<T, 4>());
    typename TTypes<U>::ConstVec scale(scale_input.vec<U>());
    typename TTypes<U>::ConstVec offset(offset_input.vec<U>());
    typename TTypes<U>::ConstVec estimated_mean(estimated_mean_input.vec<U>());
    typename TTypes<U>::ConstVec estimated_variance(
        estimated_variance_input.vec<U>());
    typename TTypes<T, 4>::Tensor y(transformed_y.tensor<T, 4>());
    typename TTypes<U>::Vec batch_mean(batch_mean_output->vec<U>());
    typename TTypes<U>::Vec batch_variance(batch_var_output->vec<U>());

    const int depth = x.dimension(3);
    const int size = x.size();
    const int rest_size = size / depth;
    Eigen::DSizes<Eigen::Index, 2> rest_by_depth(rest_size, depth);
    Eigen::DSizes<Eigen::Index, 2> one_by_depth(1, depth);

    auto x_rest_by_depth = x.reshape(rest_by_depth).template cast<U>();
    auto x_centered =
        x_rest_by_depth -
        estimated_mean.reshape(one_by_depth).broadcast(bcast_spec);
    auto scaling_factor = ((estimated_variance + epsilon).rsqrt() * scale)
                              .eval()
                              .reshape(one_by_depth)
                              .broadcast(bcast_spec);
    auto x_scaled = x_centered * scaling_factor;
    auto x_shifted =
        (x_scaled + offset.reshape(one_by_depth).broadcast(bcast_spec))
            .template cast<T>();

    y.reshape(rest_by_depth).device(d) = x_shifted;
    batch_mean.device(d) = estimated_mean;
    batch_variance.device(d) = estimated_variance;
}

} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_FILL_H__
