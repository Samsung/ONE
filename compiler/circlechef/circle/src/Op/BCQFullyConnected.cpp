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

#include "BCQFullyConnected.h"

#include "Convert.h"

namespace circlechef
{

void CircleOpBCQFullyConnected::filler(const circle::Operator *op, CircleImport *import,
                                       circlechef::ModelRecipe *model_recipe) const
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());

  import->set_tensor_filler(inputs[1]);
  import->set_tensor_filler(inputs[3]);

  const circle::Tensor *tensor2 = import->tensors()->Get(inputs[2]);
  assert(tensor2->type() == circle::TensorType::TensorType_INT32);
  const circle::Buffer *buffer2 = import->buffers()->Get(tensor2->buffer());
  auto vec2 = extract_buffer<int32_t>(buffer2);
  import->set_tensor_filler(inputs[2], vec2);

  const circle::Tensor *tensor4 = import->tensors()->Get(inputs[4]);
  assert(tensor4->type() == circle::TensorType::TensorType_INT32);
  const circle::Buffer *buffer4 = import->buffers()->Get(tensor4->buffer());
  auto vec4 = extract_buffer<int32_t>(buffer4);
  import->set_tensor_filler(inputs[4], vec4);
}

circlechef::Operation *CircleOpBCQFullyConnected::build(const circle::Operator *op,
                                                        CircleImport *import,
                                                        circlechef::ModelRecipe *model_recipe) const
{
  auto op_params = op->builtin_options_as_BCQFullyConnectedOptions();
  assert(op_params != nullptr);

  auto operation = model_recipe->add_operation();

  operation->set_type("BCQFullyConnected");

  auto op_options = operation->mutable_bcq_fully_connected_options();

  op_options->set_weights_hidden_size(op_params->weights_hidden_size());
  op_options->set_activation(as_circlechef_activation(op_params->fused_activation_function()));

  return operation;
}

} // namespace circlechef
