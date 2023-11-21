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

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

void LossCategoricalCrossentropyLayer::configure(const IPortableTensor *y_pred,
                                                 const IPortableTensor *y_true,
                                                 IPortableTensor *output,
                                                 IPortableTensor *back_prop_y_pred,
                                                 LossReductionType reduction_type, int32_t axis,
                                                 float label_smoothing)
{
  LossLayer::configure(y_pred, y_true, output, back_prop_y_pred, reduction_type);

  _axis = axis;
  _label_smoothing = label_smoothing;

  // TODO Consider broadcast
  _scratch = std::make_unique<Tensor>(_y_pred->get_info(), _y_pred->layout());
  _scratch->setBuffer(std::make_shared<basic::Allocator>(_y_pred->total_size()));
}

void LossCategoricalCrossentropyLayer::categoricalCrossEntropyFloat32()
{
  // nnfw::cker::train::CategoricalCrossEntropy(getShape(_y_pred), getBuffer<float>(_y_pred),
  //                                            getShape(_y_true), getBuffer<float>(_y_true),
  //                                            getShape(_output), getBuffer<float>(_output),
  //                                            getShape(_scratch.get()), getBuffer<float>(_scratch.get()),
  //                                            getShape(_back_prop_y_pred), getBuffer<float>(_back_prop_y_pred));
  if (getNumberOfDimensions(_y_pred) == 1)
  {
    uint32_t input_size = getNumberOfElements(_y_pred);
    nnfw::cker::train::CategoricalCrossEntropy(getBuffer<float>(_y_pred), getBuffer<float>(_y_true),
                                               getBuffer<float>(_output), 1, input_size);
  }
  else if (getNumberOfDimensions(_y_pred) == 2)
  {
    uint32_t batch_size = getSizeOfDimension(_y_pred, 0);
    if (batch_size == 0)
      throw std::runtime_error("batch_size should not be 0");

    uint32_t input_size = getNumberOfElements(_y_pred) / batch_size;
    nnfw::cker::train::CategoricalCrossEntropy(getBuffer<float>(_y_pred), getBuffer<float>(_y_true),
                                               getBuffer<float>(_output), batch_size, input_size);
  }
  else
  {
    throw std::runtime_error("LossLayer: unsupported Dimensions");
  }
}

void LossCategoricalCrossentropyLayer::categoricalCrossEntropyGradFloat32()
{
  // NOTHING TO DO
  if (getNumberOfDimensions(_y_pred) == 1)
  {
    uint32_t input_size = getNumberOfElements(_y_pred);
    nnfw::cker::train::CategoricalCrossEntropyGrad(
      getBuffer<float>(_y_pred), getBuffer<float>(_y_true), getBuffer<float>(_back_prop_y_pred), 1,
      input_size);
  }
  else if (getNumberOfDimensions(_y_pred) == 2)
  {
    uint32_t batch_size = getSizeOfDimension(_y_pred, 0);
    if (batch_size == 0)
      throw std::runtime_error("batch_size should not be 0");

    uint32_t input_size = getNumberOfElements(_y_pred) / batch_size;
    nnfw::cker::train::CategoricalCrossEntropyGrad(
      getBuffer<float>(_y_pred), getBuffer<float>(_y_true), getBuffer<float>(_back_prop_y_pred),
      batch_size, input_size);
  }
  else
  {
    throw std::runtime_error("LossLayer: unsupported Dimensions");
  }
}

void LossCategoricalCrossentropyLayer::forward(bool)
{
  if (_y_pred->data_type() == OperandType::FLOAT32)
  {
    categoricalCrossEntropyFloat32();
  }
  else
  {
    throw std::runtime_error("LossCategoricalCrossentropyLayer: unsupported data type");
  }
}

void LossCategoricalCrossentropyLayer::backward()
{
  if (_y_pred->data_type() == OperandType::FLOAT32)
  {
    categoricalCrossEntropyGradFloat32();
  }
  else
  {
    throw std::runtime_error("LossCategoricalCrossentropyLayer: unsupported data type");
  }
}

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert
