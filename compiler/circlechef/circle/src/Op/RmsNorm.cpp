/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "RmsNorm.h"

#include "Convert.h"

namespace circlechef
{

void CircleOpRmsNorm::filler(const circle::Operator *op, CircleImport *import,
                             circlechef::ModelRecipe *model_recipe) const
{
  // index 1 and 2 maybe constant
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());
  assert(inputs.size() == 3);

  import->set_tensor_filler(inputs[1]); // set gaussian filler
  import->set_tensor_filler(inputs[2]);
}

circlechef::Operation *CircleOpRmsNorm::build(const circle::Operator *op, CircleImport *import,
                                              circlechef::ModelRecipe *model_recipe) const
{
  auto operation = model_recipe->add_operation();

  operation->set_type("RmsNorm");

  auto op_options = operation->mutable_rms_norm_options();

  auto op_params = op->builtin_options_as_RmsNormOptions();
  assert(op_params != nullptr);

  op_options->set_epsilon(op_params->epsilon());

  return operation;
}

} // namespace circlechef
