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

#include "Squeeze.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpSqueeze::filler(const tflite::Operator *op, TFliteImport *import,
                             tflchef::ModelRecipe *model_recipe) const
{
  // Nothing to do with filler
}

tflchef::Operation *TFliteOpSqueeze::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_SqueezeOptions();
  assert(op_params != nullptr);

  operation->set_type("Squeeze");

  auto op_options = operation->mutable_squeeze_options();

  std::vector<int32_t> squeeze_dims = as_index_vector(op_params->squeeze_dims());

  for (auto dim : squeeze_dims)
  {
    op_options->add_squeeze_dim(dim);
  }

  return operation;
}

} // namespace tflchef
