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

#include "Cast.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpCast::filler(const tflite::Operator *op, TFliteImport *import,
                          tflchef::ModelRecipe *model_recipe) const
{
  // Nothing to do with filler
}

tflchef::Operation *TFliteOpCast::build(RecipeChefContext *ctx) const

{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_CastOptions();
  assert(op_params != nullptr);

  operation->set_type("Cast");

  auto op_options = operation->mutable_cast_options();

  op_options->set_in_data_type(as_tflchef_type(op_params->in_data_type()));
  op_options->set_out_data_type(as_tflchef_type(op_params->out_data_type()));

  return operation;
}

} // namespace tflchef
