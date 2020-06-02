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

#include "DepthToSpace.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpDepthToSpace::filler(const tflite::Operator *op, TFliteImport *import,
                                  tflchef::ModelRecipe *model_recipe) const
{
  // Nothing to do with filler
}

tflchef::Operation *TFliteOpDepthToSpace::build(const tflite::Operator *op, TFliteImport *import,
                                                tflchef::ModelRecipe *model_recipe) const
{
  auto operation = model_recipe->add_operation();

  operation->set_type("DepthToSpace");

  auto op_params = op->builtin_options_as_DepthToSpaceOptions();
  assert(op_params != nullptr);

  auto op_options = operation->mutable_depth_to_space_options();

  op_options->set_block_size(op_params->block_size());

  return operation;
}

} // namespace tflchef
