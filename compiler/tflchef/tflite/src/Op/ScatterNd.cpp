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

#include "ScatterNd.h"

#include "Convert.h"
#include "FillerHelper.h"

namespace tflchef
{

void TFliteOpScatterNd::filler(const tflite::Operator *op, TFliteImport *import,
                               tflchef::ModelRecipe *model_recipe) const
{
  // Filler for indices and shape
  fill_tensor_to_import(0, import);
  fill_tensor_to_import(2, import);
}

tflchef::Operation *TFliteOpScatterNd::build(const tflite::Operator *, TFliteImport *,
                                             tflchef::ModelRecipe *model_recipe) const
{
  auto operation = model_recipe->add_operation();

  operation->set_type("ScatterNd");

  return operation;
}

} // namespace tflchef
