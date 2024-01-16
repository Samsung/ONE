/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CirGru.h"

#include "Convert.h"

namespace circlechef
{

void CircleOpCirGru::filler(const circle::Operator *op, CircleImport *import,
                            circlechef::ModelRecipe *model_recipe) const
{
  // index 1, 2, 3, 4, 5 maybe constant
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());
  assert(inputs.size() == 6);

  import->set_tensor_filler(inputs[1]); // set gaussian filler
  import->set_tensor_filler(inputs[2]);
  import->set_tensor_filler(inputs[3]);
  import->set_tensor_filler(inputs[4]);
  import->set_tensor_filler(inputs[5]);
}

circlechef::Operation *CircleOpCirGru::build(const circle::Operator *op, CircleImport *import,
                                             circlechef::ModelRecipe *model_recipe) const
{
  auto op_params = op->builtin_options_as_CirGruOptions();
  assert(op_params != nullptr);

  auto operation = model_recipe->add_operation();

  operation->set_type("CirGru");

  auto op_options = operation->mutable_circle_gru_options();

  op_options->set_activation(as_circlechef_activation(op_params->fused_activation_function()));
  op_options->set_return_sequences(op_params->return_sequences());
  op_options->set_time_major(op_params->time_major());

  return operation;
}

} // namespace circlechef
