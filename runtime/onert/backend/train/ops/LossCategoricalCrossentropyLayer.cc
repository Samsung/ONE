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

#include "LossCategoricalCrossentropyLayer.h"
#include "OperationUtils.h"

#include <cker/train/operation/Loss.h>
#include <util/Exceptions.h>

namespace onert::backend::train::ops
{

void LossCategoricalCrossentropyLayer::configure(
  const IPortableTensor *y_pred, const IPortableTensor *y_true, IPortableTensor *output,
  IPortableTensor *back_prop_y_pred, ir::train::LossReductionType reduction_type, int32_t axis,
  float label_smoothing, bool is_normalization_required)
{
  LossLayer::configure(y_pred, y_true, output, back_prop_y_pred, reduction_type);

  _axis = axis;
  _label_smoothing = label_smoothing;
  _is_normalization_required = is_normalization_required;
}

void LossCategoricalCrossentropyLayer::forward(bool)
{
  if (_y_pred->data_type() == OperandType::FLOAT32)
  {
    nnfw::cker::train::CategoricalCrossEntropy(getShape(_y_pred), getBuffer<float>(_y_pred),
                                               getShape(_y_true), getBuffer<float>(_y_true),
                                               getShape(_output), getBuffer<float>(_output));
  }
  else
  {
    throw UnsupportedDataTypeException{"train LossCategoricalCrossEntropy", _y_pred->data_type()};
  }
}

void LossCategoricalCrossentropyLayer::backward()
{
  assert(_back_prop_y_pred != nullptr);

  if (_y_pred->data_type() == OperandType::FLOAT32)
  {
    const auto reduction_type = convertLossReductionType(_reduction_type);
    if (_is_normalization_required)
    {
      // TODO Eliminate duplicate calculations for output
      nnfw::cker::train::CategoricalCrossEntropyWithLogits(
        getShape(_y_pred), getBuffer<float>(_y_pred), getShape(_y_true), getBuffer<float>(_y_true),
        getShape(_output), getBuffer<float>(_output), getShape(_back_prop_y_pred),
        getBuffer<float>(_back_prop_y_pred), reduction_type);
    }
    else
    {
      nnfw::cker::train::CategoricalCrossEntropyGrad(
        getShape(_y_pred), getBuffer<float>(_y_pred), getShape(_y_true), getBuffer<float>(_y_true),
        getShape(_back_prop_y_pred), getBuffer<float>(_back_prop_y_pred), reduction_type);
    }
  }
  else
  {
    throw UnsupportedDataTypeException{"train LossCategoricalCrossEntropy", _y_pred->data_type()};
  }
}

} // namespace onert::backend::train::ops
