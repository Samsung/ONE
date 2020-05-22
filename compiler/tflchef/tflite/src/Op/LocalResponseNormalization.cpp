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

#include "LocalResponseNormalization.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpLocalResponseNormalization::filler(const tflite::Operator *op, TFliteImport *import,
                                                tflchef::ModelRecipe *model_recipe) const
{
  // Nothing to do with filler
}

tflchef::Operation *
TFliteOpLocalResponseNormalization::build(const tflite::Operator *op, TFliteImport *import,
                                          tflchef::ModelRecipe *model_recipe) const
{
  auto op_params = op->builtin_options_as_LocalResponseNormalizationOptions();
  assert(op_params != nullptr);

  auto operation = model_recipe->add_operation();

  operation->set_type("LocalResponseNormalization");

  auto op_options = operation->mutable_local_response_normalization_options();

  op_options->set_radius(op_params->radius());
  op_options->set_bias(op_params->bias());
  op_options->set_alpha(op_params->alpha());
  op_options->set_beta(op_params->beta());

  return operation;
}

} // namespace tflchef
