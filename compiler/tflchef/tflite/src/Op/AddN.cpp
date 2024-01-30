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

#include "AddN.h"

#include "Convert.h"
#include "FillerHelper.h"

namespace tflchef
{

void TFliteOpAddN::filler(const tflite::Operator *op, TFliteImport *import,
                          tflchef::ModelRecipe *model_recipe) const
{
  // AddN may have constant input

  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());

  for (uint32_t idx = 0; idx < inputs.size(); ++idx)
    fill_tensor_to_import(inputs[idx], import);
}

tflchef::Operation *TFliteOpAddN::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  operation->set_type("AddN");

  return operation;
}

} // namespace tflchef
