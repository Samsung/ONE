/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DepthwiseConv2D.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpDepthwiseConv2D::filler(const tflite::Operator *op, TFliteImport *import,
                                     tflchef::ModelRecipe *model_recipe) const
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());

  bool hasBias = (inputs.size() == 3);
  assert(inputs.size() == 2 || hasBias);

  import->set_tensor_filler(inputs.at(1)); // kernel
  if (hasBias)
    import->set_tensor_filler(inputs.at(2)); // bias
}

tflchef::Operation *TFliteOpDepthwiseConv2D::build(RecipeChefContext *ctx) const

{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_DepthwiseConv2DOptions();
  assert(op_params != nullptr);

  operation->set_type("DepthwiseConv2D");

  auto op_options = operation->mutable_depthwiseconv2d_options();

  op_options->set_activation(as_tflchef_activation(op_params->fused_activation_function()));
  op_options->set_stride_h(op_params->stride_h());
  op_options->set_stride_w(op_params->stride_w());
  op_options->set_depth_multiplier(op_params->depth_multiplier());
  op_options->set_dilation_w_factor(op_params->dilation_w_factor());
  op_options->set_dilation_h_factor(op_params->dilation_h_factor());
  op_options->set_padding(as_tflchef_padding(op_params->padding()));

  return operation;
}

} // namespace tflchef
