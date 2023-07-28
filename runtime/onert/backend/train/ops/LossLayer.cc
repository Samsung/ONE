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
#include "OperationUtils.h"

#include <cker/train/operation/Loss.h>
#include <cker/train/operation/Regularizer.h>

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
    _loss_type(LossType::kMSE)
{
  // DO NOTHING
}

void LossLayer::configure(const IPortableTensor *y_pred, const IPortableTensor *y_true,
                          IPortableTensor *output, IPortableTensor *deriv_y_pred,
                          LossType loss_type)
{
  assert(y_pred != nullptr);
  assert(y_true != nullptr);
  assert(output != nullptr);
  assert(deriv_y_pred != nullptr);
  assert(output->data_type() == OperandType::FLOAT32);

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
  _loss_type = loss_type;
}

void LossLayer::forward(bool)
{
  // TODO Implement this
  switch (_loss_type)
  {
    case LossType::kMSE:
      if (_y_pred->data_type() == OperandType::FLOAT32)
      {
        nnfw::cker::train::MSE(getShape(_y_pred), getBuffer<float>(_y_pred), getShape(_y_true),
                               getBuffer<float>(_y_true), getShape(_output),
                               getBuffer<float>(_output));
      }
      break;
    default:
      throw std::runtime_error("LossLayer: unsupported loss type");
  }

  // nnfw::cker::train::L2(getShape(_output), getBuffer<float>(_output), 0.01f);
}

void LossLayer::backward()
{
  switch (_loss_type)
  {
    case LossType::kMSE:
      if (_y_pred->data_type() == OperandType::FLOAT32)
      {
        nnfw::cker::train::MSEGrad(getShape(_y_pred), getBuffer<float>(_y_pred), getShape(_y_true),
                                   getBuffer<float>(_y_true), getShape(_deriv_y_pred),
                                   getBuffer<float>(_deriv_y_pred));
      }
      break;
    default:
      throw std::runtime_error("LossLayer: unsupported loss type");
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
