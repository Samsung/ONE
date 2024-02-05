/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "GradientAppender.h"

#include "OperationUtils.h"

#include <cker/operation/BinaryArithmeticOps.h>
#include <util/CalculateActivationRange.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

GradientAppender::GradientAppender(const IPortableTensor *temp_tensor,
                                   IPortableTensor *gradient_tensor)
  : _temp_tensor{temp_tensor}, _gradient_tensor{gradient_tensor}
{
  if (temp_tensor->getShape() != gradient_tensor->getShape())
    throw std::runtime_error("train GradientAppender: Unsupported shapes");

  float output_activation_min = 0, output_activation_max = 0;
  util::CalculateActivationRange(ir::Activation::NONE, &output_activation_min,
                                 &output_activation_max);
  _op_params.float_activation_max = output_activation_max;
  _op_params.float_activation_min = output_activation_min;
}

void GradientAppender::forward(bool)
{
  // DO NOTHING
}

void GradientAppender::backward()
{
  nnfw::cker::BinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::ADD>(
    _op_params, getShape(_temp_tensor), getBuffer<float>(_temp_tensor), getShape(_gradient_tensor),
    getBuffer<float>(_gradient_tensor), getShape(_gradient_tensor),
    getBuffer<float>(_gradient_tensor));
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
