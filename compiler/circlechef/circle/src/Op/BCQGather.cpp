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

#include "BCQGather.h"

#include "Convert.h"

namespace circlechef
{

void CircleOpBCQGather::filler(const circle::Operator *op, CircleImport *import,
                               circlechef::ModelRecipe *model_recipe) const
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());

  import->set_tensor_filler(inputs[0]);

  const circle::Tensor *tensor1 = import->tensors()->Get(inputs[1]);
  assert(tensor1->type() == circle::TensorType::TensorType_INT32);
  const circle::Buffer *buffer1 = import->buffers()->Get(tensor1->buffer());
  auto vec1 = extract_buffer<int32_t>(buffer1);
  import->set_tensor_filler(inputs[1], vec1);

  const circle::Tensor *tensor2 = import->tensors()->Get(inputs[2]);
  assert(tensor2->type() == circle::TensorType::TensorType_INT32);
  const circle::Buffer *buffer2 = import->buffers()->Get(tensor2->buffer());
  auto vec2 = extract_buffer<int32_t>(buffer2);
  import->set_tensor_filler(inputs[2], vec2);

  const circle::Tensor *tensor3 = import->tensors()->Get(inputs[3]);
  assert(tensor3->type() == circle::TensorType::TensorType_INT32);
  const circle::Buffer *buffer3 = import->buffers()->Get(tensor3->buffer());
  auto vec3 = extract_buffer<int32_t>(buffer3);
  import->set_tensor_filler(inputs[3], vec3);
}

circlechef::Operation *CircleOpBCQGather::build(const circle::Operator *op, CircleImport *import,
                                                circlechef::ModelRecipe *model_recipe) const
{
  auto op_params = op->builtin_options_as_BCQGatherOptions();
  assert(op_params != nullptr);

  auto operation = model_recipe->add_operation();

  operation->set_type("BCQGather");

  auto op_options = operation->mutable_bcq_gather_options();

  op_options->set_input_hidden_size(op_params->input_hidden_size());
  op_options->set_axis(op_params->axis());

  return operation;
}

} // namespace circlechef
