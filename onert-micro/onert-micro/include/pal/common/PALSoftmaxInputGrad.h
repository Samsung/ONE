/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ONERT_MICRO_EXECUTE_PAL_COMMON_SOFTMAX_INPUT_GRAD_H
#define ONERT_MICRO_EXECUTE_PAL_COMMON_SOFTMAX_INPUT_GRAD_H

#include "OMStatus.h"
#include "PALUtils.h"

#include <cmath>

namespace onert_micro
{
namespace train
{
namespace pal
{

void inline SoftmaxInputGrad(const float *dloss_doutput_data,
                             const core::OMRuntimeShape &dloss_doutput_shape,
                             const float *calculated_data, float *jacobian_row_data,
                             float *dloss_dinput_data)
{
  assert(dloss_doutput_shape.dimensionsCount() == 2);
  assert(dloss_doutput_shape.dims(0) == 1);
  const uint32_t output_dim = dloss_doutput_shape.dims(dloss_doutput_shape.dimensionsCount() - 1);
  for (int i = 0; i < output_dim; ++i)
  {
    for (int j = 0; j < output_dim; ++j)
    {
      jacobian_row_data[j] = -calculated_data[i] * calculated_data[j];
    }
    jacobian_row_data[i] += calculated_data[i];
    float total = 0.f;
    for (int j = 0; j < output_dim; ++j)
    {
      total += jacobian_row_data[j] * dloss_doutput_data[j];
    }
    dloss_dinput_data[i] = total;
  }
}

} // namespace pal
} // namespace train
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_PAL_COMMON_SOFTMAX_INPUT_GRAD_H
