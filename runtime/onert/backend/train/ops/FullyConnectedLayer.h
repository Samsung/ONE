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

#ifndef __ONERT_BACKEND_TRAIN_OPS_FULLYCONNECTEDLAYER_H__
#define __ONERT_BACKEND_TRAIN_OPS_FULLYCONNECTEDLAYER_H__

#include "../ExternalContext.h"
#include "../Tensor.h"

#include <exec/train/ITrainableFunction.h>
#include <ops/FullyConnectedLayer.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

class FullyConnectedLayer : public exec::train::ITrainableFunction,
                            public cpu::ops::FullyConnectedLayer
{
public:
  FullyConnectedLayer();
  ~FullyConnectedLayer();

public:
  void configure(const IPortableTensor *input, const IPortableTensor *weights,
                 const IPortableTensor *bias, IPortableTensor *output, IPortableTensor *deriv_input,
                 IPortableTensor *grad_weights, IPortableTensor *grad_bias,
                 const IPortableTensor *deriv_output, ir::Activation activation,
                 ir::FullyConnectedWeightsFormat weights_format,
                 const std::shared_ptr<train::ExternalContext> &external_context);

  void forward(bool training) override;
  void backward(uint32_t training_step) override;

private:
  void backwardFloat32();

private:
  IPortableTensor *_grad_weights;
  IPortableTensor *_grad_bias;
  IPortableTensor *_deriv_input;
  const IPortableTensor *_deriv_output;

  // TODO Optimize memory
  std::unique_ptr<Tensor> _transposed_weights;
  std::unique_ptr<Tensor> _transposed_input;
  std::unique_ptr<Tensor> _transposed_deriv_output;
  std::unique_ptr<Tensor> _act_deriv_output;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_FULLYCONNECTEDLAYER_H__
