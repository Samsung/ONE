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

#ifndef __ONERT_BACKEND_TRAIN_OPS_LOSS_CATEGORICALCROSSENTROPY_LAYER_H__
#define __ONERT_BACKEND_TRAIN_OPS_LOSS_CATEGORICALCROSSENTROPY_LAYER_H__

#include "LossLayer.h"
#include "../Tensor.h"

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

class LossCategoricalCrossentropyLayer : public LossLayer
{
public:
  LossCategoricalCrossentropyLayer() = default;

  void configure(const IPortableTensor *y_pred, const IPortableTensor *y_true,
                 IPortableTensor *output, IPortableTensor *back_prop_y_pred,
                 LossReductionType reduction_type, int32_t axis, float label_smoothing);
  void forward(bool training) override;
  void backward() override;

private:
  void categoricalCrossEntropyFloat32();
  void categoricalCrossEntropyGradFloat32();

private:
  int32_t _axis;
  float _label_smoothing;

  // TODO Consider if these tensors should be built in TensorBuilder
  std::unique_ptr<Tensor> _scratch;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_LOSS_CATEGORICALCROSSENTROPY_LAYER_H__
