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

#include "Sub.h"

#include "Convert.h"
#include "FillerHelper.h"

namespace tflchef
{

void TFliteOpSub::filler(const tflite::Operator *op, TFliteImport *import,
                         tflchef::ModelRecipe *model_recipe) const
{
  // Sub may have constant input

  fill_two_inputs(op, import);
}

tflchef::Operation *TFliteOpSub::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_SubOptions();
  assert(op_params != nullptr);

  operation->set_type("Sub");

  auto op_options = operation->mutable_sub_options();

  auto tflchef_activation = as_tflchef_activation(op_params->fused_activation_function());
  op_options->set_activation(tflchef_activation);

  return operation;
}

} // namespace tflchef
