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

#include "Adam.h"

#include "../ops/OperationUtils.h"
#include <cker/train/optimizer/Adam.h>
#include <cmath>
#include <misc/polymorphic_downcast.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace optimizer
{

double Adam::getLearningRate(uint32_t training_step) const
{
  auto biasCorrection = [&](double f) { return 1.0f - std::pow(f, training_step + 1); };
  return _learning_rate * (std::sqrt(biasCorrection(_props.beta2)) / biasCorrection(_props.beta1));
}

void Adam::applyGradient(const UpdateFactors &factors) const
{
  const auto training_step = std::get<size_t>(factors);
  const auto &grad_tensor = std::get<const backend::IPortableTensor &>(factors);
  auto &trainable_tensor = std::get<backend::train::ITrainableTensor &>(factors);
  assert(trainable_tensor.data_type() == grad_tensor.data_type());
  const auto opt_vars = trainable_tensor.optVars();
  assert(opt_vars.size() == 2);
  // Get the variable for exponential moving average of the gradient
  auto m_tensor = nnfw::misc::polymorphic_downcast<IPortableTensor *>(opt_vars.at(0));
  // Get the variable for exponential moving average of the squared_gradient
  auto v_tensor = nnfw::misc::polymorphic_downcast<IPortableTensor *>(opt_vars.at(1));

  const auto beta1_power = std::pow(_props.beta1, training_step + 1);
  const auto beta2_power = std::pow(_props.beta2, training_step + 1);
  // TODO Support nesterov
  const bool use_nesterov = false;

  if (trainable_tensor.getShape() != grad_tensor.getShape())
  {
    throw std::runtime_error("Adam: Invalid gradient tensor");
  }

  switch (grad_tensor.data_type())
  {
    case ir::DataType::FLOAT32:
      nnfw::cker::train::Adam(
        ops::getShape(&trainable_tensor), ops::getBuffer<float>(&trainable_tensor),
        ops::getShape(&grad_tensor), ops::getBuffer<float>(&grad_tensor), ops::getShape(m_tensor),
        ops::getBuffer<float>(m_tensor), ops::getShape(v_tensor), ops::getBuffer<float>(v_tensor),
        beta1_power, beta2_power, _learning_rate, _props.beta1, _props.beta2, _props.epsilon,
        use_nesterov);
      break;
    default:
      throw std::runtime_error("Adam: Not supported data type");
  }
}

} // namespace optimizer
} // namespace train
} // namespace backend
} // namespace onert
