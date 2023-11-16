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

#include "LossMeanSquaredErrorLayer.h"
#include "OperationUtils.h"

#include <cker/train/operation/Loss.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

void LossMeanSquaredErrorLayer::configure(const IPortableTensor *y_pred,
                                          const IPortableTensor *y_true, IPortableTensor *output,
                                          IPortableTensor *back_prop_y_pred)
{
  LossLayer::configure(y_pred, y_true, output, back_prop_y_pred);
}

void LossMeanSquaredErrorLayer::forward(bool)
{
  // TODO Implement this
  if (_y_pred->data_type() == OperandType::FLOAT32)
  {
    nnfw::cker::train::MSE(getShape(_y_pred), getBuffer<float>(_y_pred), getShape(_y_true),
                           getBuffer<float>(_y_true), getShape(_output), getBuffer<float>(_output));
  }
  else
  {
    throw std::runtime_error("LossMeanSquaredErrorLayer: unsupported data type");
  }
}

void LossMeanSquaredErrorLayer::backward()
{
  if (_y_pred->data_type() == OperandType::FLOAT32)
  {
    nnfw::cker::train::MSEGrad(getShape(_y_pred), getBuffer<float>(_y_pred), getShape(_y_true),
                               getBuffer<float>(_y_true), getShape(_back_prop_y_pred),
                               getBuffer<float>(_back_prop_y_pred));
  }
  else
  {
    throw std::runtime_error("LossMeanSquaredErrorLayer: unsupported data type");
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
