/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_TRAIN_OPS_DEPTHWISECONVOLUTIONLAYER_H__
#define __ONERT_BACKEND_TRAIN_OPS_DEPTHWISECONVOLUTIONLAYER_H__

#include <ops/DepthwiseConv2DLayer.h>
#include <backend/basic/Allocator.h>

#include "../Tensor.h"
#include "../ExternalContext.h"
#include <exec/train/ITrainableFunction.h>

namespace onert::backend::train::ops
{

class DepthwiseConvolutionLayer : public ::onert::exec::train::ITrainableFunction,
                                  public cpu::ops::DepthwiseConvolutionLayer
{
public:
  DepthwiseConvolutionLayer();

  void configureBackward(IPortableTensor *back_prop_input, IPortableTensor *grad_weights,
                         IPortableTensor *grad_bias, const IPortableTensor *back_prop_output,
                         const ir::Activation activation);
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
  std::unique_ptr<BackPropTensor> _act_back_prop_output;

  bool _use_padded_filter;
  std::unique_ptr<Tensor> _padded_filter;
  std::unique_ptr<Tensor> _filter_buffers;
  std::unique_ptr<Tensor> _filter_dim_buffers;
};

} // namespace onert::backend::train::ops

#endif // __ONERT_BACKEND_TRAIN_OPS_DEPTHWISECONVOLUTIONLAYER_H__
