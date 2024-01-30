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

#include "BatchMatMul.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpBatchMatMul::filler(const tflite::Operator *op, TFliteImport *import,
                                 tflchef::ModelRecipe *model_recipe) const
{
  // Nothing to do with filler
}

tflchef::Operation *TFliteOpBatchMatMul::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  operation->set_type("BatchMatMul");

  auto op_options = operation->mutable_batch_matmul_options();

  auto op_params = op->builtin_options_as_BatchMatMulOptions();
  assert(op_params != nullptr);

  op_options->set_adj_x(op_params->adj_x());
  op_options->set_adj_y(op_params->adj_y());

  return operation;
}

} // namespace tflchef
