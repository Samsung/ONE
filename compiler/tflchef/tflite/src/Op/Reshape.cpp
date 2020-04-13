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

#include "Reshape.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpReshape::filler(const tflite::Operator *op, TFliteImport *import,
                             tflchef::ModelRecipe *model_recipe) const
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());

  bool hasShape = (inputs.size() == 2);
  assert(inputs.size() == 1 || hasShape);

  if (hasShape)
  {
    auto op_params = op->builtin_options_as_ReshapeOptions();
    std::vector<int32_t> new_shape = as_index_vector(op_params->new_shape());
    import->set_tensor_filler(inputs.at(1), new_shape);
  }
}

tflchef::Operation *TFliteOpReshape::build(const tflite::Operator *op, TFliteImport *import,
                                           tflchef::ModelRecipe *model_recipe) const
{
  auto op_params = op->builtin_options_as_ReshapeOptions();
  assert(op_params != nullptr);

  auto operation = model_recipe->add_operation();

  operation->set_type("Reshape");

  auto op_options = operation->mutable_reshape_options();

  std::vector<int32_t> new_shape = as_index_vector(op_params->new_shape());

  for (auto shape : new_shape)
  {
    op_options->add_new_shape(shape);
  }

  return operation;
}

} // namespace tflchef
