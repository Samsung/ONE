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

#include <exec/train/optimizer/Adam.h>

#include <exec/train/optimizer/OptimizerHelpers.h>

#include <cmath>

namespace onert
{
namespace exec
{
namespace train
{
namespace optimizer
{

double Adam::getLearningRate(uint32_t iteration) const
{
  auto biasCorrection = [&](double f) { return 1.0f - std::pow(f, iteration + 1); };
  return _learning_rate * (std::sqrt(biasCorrection(_props.beta2)) / biasCorrection(_props.beta1));
}

void Adam::applyGradient(const UpdateFactors &factors) const
{
  const auto lr = getLearningRate(std::get<size_t>(factors));
  const auto &grad_tensor = std::get<const backend::IPortableTensor &>(factors);
  auto &trainable_tensor = std::get<backend::train::ITrainableTensor &>(factors);
  assert(trainable_tensor.data_type() == grad_tensor.data_type());


  auto &beta1 = _props.beta1;
  auto &beta2 = _props.beta2;
  auto &epsilon = _props.epsilon;

  //////////////////////////////////////////////////////
  // This is implementation of adam from original paper.

  auto opt_vars = trainable_tensor.optVars();
  assert(opt_vars.size() == 2);
  // Get the variable for exponential moving average of the gradient
  auto vv = opt_vars.at(0);
  // Get the variable for exponential moving average of the squared_gradient
  auto vs = opt_vars.at(1);

  // Update exponential moving average of the gradient at the current step
  const auto &shape = grad_tensor.get_info().shape();
  assert(vv->getShape() == shape);
  elementwise<float>(shape, grad_tensor, *vv, [&](double grad_val, double v_val) -> double {
    return beta1 * v_val + (1.0f - beta1) * grad_val;
  });

  // Update exponential moving average of the squared_gradient at the current step
  assert(vs->getShape() == shape);
  elementwise<float>(shape, grad_tensor, *vs, [&](double grad_val, double s_val) -> double {
    return beta2 * s_val + (1.0f - beta2) * (grad_val * grad_val);
  });

  // Apply gradient
  auto vBiasCorrection = [beta1](double f) { return f / (1 - beta1); };
  auto sBiasCorrection = [beta2](double f) { return f / (1 - beta2); };
  auto rescaleGrad = [&](double v, double s) {
    return (lr * vBiasCorrection(v)) / (std::sqrt(sBiasCorrection(s)) + epsilon);
  };
  ShapeLoop(shape, [&](const ir::Coordinates &coords) {
    const float v = *reinterpret_cast<const float *>(vv->buffer() + vv->calcOffset(coords));
    const float s = *reinterpret_cast<const float *>(vs->buffer() + vs->calcOffset(coords));
    float *w_data =
      reinterpret_cast<float *>(trainable_tensor.buffer() + trainable_tensor.calcOffset(coords));
    *w_data -= rescaleGrad(v, s);
  });
}

} // namespace optimizer
} // namespace train
} // namespace exec
} // namespace onert
