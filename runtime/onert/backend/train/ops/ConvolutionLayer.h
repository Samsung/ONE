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

#ifndef __ONERT_BACKEND_TRAIN_OPS_CONVOLUTIONLAYER_H__
#define __ONERT_BACKEND_TRAIN_OPS_CONVOLUTIONLAYER_H__

#include <ops/ConvolutionLayer.h>

#include "../Tensor.h"
#include <exec/train/ITrainableFunction.h>

namespace onert
{
namespace backend
{
namespace train
{
namespace ops
{

class ConvolutionLayer : public ::onert::exec::train::ITrainableFunction,
                         public cpu::ops::ConvolutionLayer
{
public:
  ConvolutionLayer();
  ~ConvolutionLayer();

  void configureBackward(const IPortableTensor *weights, IPortableTensor *back_prop_input,
                         IPortableTensor *grad_weights, IPortableTensor *grad_bias,
                         const IPortableTensor *back_prop_output, const ir::Activation activation);
  std::optional<ExtraTensors> registerExtraTensors() override;
  void forward(bool training) override;
  void backward() override;

private:
  void backwardFloat32();

private:
  IPortableTensor *_grad_weights;
  IPortableTensor *_grad_bias;
  IPortableTensor *_back_prop_input;
  const IPortableTensor *_back_prop_output;

  // TODO Consider if these tensors should be built in TensorBuilder
  std::shared_ptr<ExtraTensor> _transposed_weights;
  std::shared_ptr<ExtraTensor> _conv_back_prop_output;
  std::shared_ptr<ExtraTensor> _transposed_grad_weights;
  std::shared_ptr<ExtraTensor> _act_back_prop_output;
};

} // namespace ops
} // namespace train
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRAIN_OPS_CONVOLUTIONLAYER_H__
