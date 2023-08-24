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

#include "SGD.h"

#include "../ops/OperationUtils.h"
#include <cker/train/optimizer/SGD.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace optimizer
{

double SGD::getLearningRate(uint32_t) const
{
  // TODO Use iteration, momentum, and nesterov
  return _learning_rate;
}

void SGD::applyGradient(const UpdateFactors &factors) const
{
  const auto lr = getLearningRate(std::get<size_t>(factors));
  const auto &grad_tensor = std::get<const backend::IPortableTensor &>(factors);
  auto &trainable_tensor = std::get<backend::train::ITrainableTensor &>(factors);
  assert(trainable_tensor.data_type() == grad_tensor.data_type());

  // TODO Support for different shapes
  if (trainable_tensor.getShape() != grad_tensor.getShape())
  {
    throw std::runtime_error("SGD: Invalid gradient tensor");
  }

  switch (grad_tensor.data_type())
  {
    case ir::DataType::FLOAT32:
      nnfw::cker::train::GradientDescent(
        ops::getShape(&trainable_tensor), ops::getBuffer<float>(&trainable_tensor),
        ops::getShape(&grad_tensor), ops::getBuffer<float>(&grad_tensor), lr);
      break;
    default:
      throw std::runtime_error("SGD: Not supported data type");
  }
}

} // namespace optimizer
} // namespace train
} // namespace backend
} // namespace onert
