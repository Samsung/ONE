/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SVDF.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpSVDF::filler(const tflite::Operator *op, TFliteImport *import,
                          tflchef::ModelRecipe *model_recipe) const
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());
  assert(inputs.size() == 5);

  // optional input tensor idx has minus value.
  const bool hasBias = (inputs.at(3) >= 0);

  // Note: last input is variable tensor without data
  import->set_tensor_filler(inputs.at(1));
  import->set_tensor_filler(inputs.at(2));
  if (hasBias)
    import->set_tensor_filler(inputs.at(3));
}

tflchef::Operation *TFliteOpSVDF::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  const auto op_params = op->builtin_options_as_SVDFOptions();
  assert(op_params != nullptr);

  operation->set_type("SVDF");

  auto op_options = operation->mutable_svdf_options();

  op_options->set_activation(as_tflchef_activation(op_params->fused_activation_function()));
  op_options->set_asymmetric_quantize_inputs(op_params->asymmetric_quantize_inputs());
  op_options->set_rank(op_params->rank());

  return operation;
}

} // namespace tflchef
