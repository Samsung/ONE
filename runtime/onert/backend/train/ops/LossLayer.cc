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

#include "LossLayer.h"

#include <ops/OperationUtils.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

LossLayer::LossLayer()
  : _y_pred(nullptr), _y_true(nullptr), _output(nullptr), _deriv_y_pred(nullptr),
    _deriv_y_true(nullptr), _deriv_output(nullptr), _loss_type(LossType::kMSE)
{
  // DO NOTHING
}

void LossLayer::configure(const IPortableTensor *y_pred, const IPortableTensor *y_true,
                          IPortableTensor *output, IPortableTensor *deriv_y_pred,
                          IPortableTensor *deriv_y_true, const IPortableTensor *deriv_output,
                          LossType loss_type)
{
  assert(y_pred != nullptr);
  assert(y_true != nullptr);
  assert(output != nullptr);
  assert(deriv_y_pred != nullptr);
  assert(deriv_y_true != nullptr);
  assert(deriv_output != nullptr);

  switch (loss_type)
  {
    case LossType::kMSE:
      break;
    default:
      throw std::runtime_error("LossLayer: unsupported loss type");
  }

  _y_pred = y_pred;
  _y_true = y_true;
  _output = output;
  _deriv_y_pred = deriv_y_pred;
  _deriv_y_true = deriv_y_true;
  _deriv_output = deriv_output;
  _loss_type = loss_type;
}

void LossLayer::forward(bool)
{
  // TODO Implement this
}

void LossLayer::backward(uint32_t)
{
  // TODO Implement this
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
