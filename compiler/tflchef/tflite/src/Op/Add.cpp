/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Add.h"

#include "Convert.h"
#include "FillerHelper.h"

namespace tflchef
{

void TFliteOpAdd::filler(const tflite::Operator *op, TFliteImport *import,
                         tflchef::ModelRecipe *model_recipe) const
{
  // Add may have constant input

  fill_two_inputs(op, import);
}

tflchef::Operation *TFliteOpAdd::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_AddOptions();
  assert(op_params != nullptr);

  operation->set_type("Add");

  auto op_options = operation->mutable_add_options();

  op_options->set_activation(as_tflchef_activation(op_params->fused_activation_function()));

  return operation;
}

} // namespace tflchef
