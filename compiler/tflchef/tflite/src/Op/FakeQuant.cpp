/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "FakeQuant.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpFakeQuant::filler(const tflite::Operator *op, TFliteImport *import,
                               tflchef::ModelRecipe *model_recipe) const
{
  // Nothing to do with filler
}

tflchef::Operation *TFliteOpFakeQuant::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_FakeQuantOptions();
  assert(op_params != nullptr);

  operation->set_type("FakeQuant");

  auto op_options = operation->mutable_fakequant_options();

  op_options->set_min(op_params->min());
  op_options->set_max(op_params->max());
  op_options->set_num_bits(op_params->num_bits());
  op_options->set_narrow_range(op_params->narrow_range());

  return operation;
}

} // namespace tflchef
