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

#include "MaxPool2D.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpMaxPool2D::filler(const tflite::Operator *op, TFliteImport *import,
                               tflchef::ModelRecipe *model_recipe) const
{
  // Nothing to do with filler
}

tflchef::Operation *TFliteOpMaxPool2D::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_Pool2DOptions();
  assert(op_params != nullptr);

  operation->set_type("MaxPool2D");

  auto op_options = operation->mutable_maxpool2d_options();

  op_options->set_padding(as_tflchef_padding(op_params->padding()));
  op_options->set_stride_h(op_params->stride_h());
  op_options->set_stride_w(op_params->stride_w());
  op_options->set_filter_height(op_params->filter_height());
  op_options->set_filter_width(op_params->filter_width());
  op_options->set_activation(as_tflchef_activation(op_params->fused_activation_function()));

  return operation;
}

} // namespace tflchef
