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

#include "Softmax.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpSoftmax::filler(const tflite::Operator *op, TFliteImport *import,
                             tflchef::ModelRecipe *model_recipe) const
{
  // Nothing to do with filler
}

tflchef::Operation *TFliteOpSoftmax::build(RecipeChefContext *ctx) const

{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_SoftmaxOptions();
  assert(op_params != nullptr);

  operation->set_type("Softmax");

  auto op_options = operation->mutable_softmax_options();

  op_options->set_beta(op_params->beta());

  return operation;
}

} // namespace tflchef
