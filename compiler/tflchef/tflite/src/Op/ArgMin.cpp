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

#include "ArgMin.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpArgMin::filler(const tflite::Operator *op, TFliteImport *import,
                            tflchef::ModelRecipe *model_recipe) const
{
  // filler for second input, argmin/dim
  const auto &inputs = *op->inputs();

  const tflite::Tensor *dim_tensor = import->tensors()->Get(inputs[1]);
  assert(dim_tensor->type() == tflite::TensorType::TensorType_INT32);
  const tflite::Buffer *buffer = import->buffers()->Get(dim_tensor->buffer());
  auto vec = extract_buffer<int32_t>(buffer);
  import->set_tensor_filler(inputs[1], vec);
}

tflchef::Operation *TFliteOpArgMin::build(const tflite::Operator *op, TFliteImport *import,
                                          tflchef::ModelRecipe *model_recipe) const
{
  auto op_params = op->builtin_options_as_ArgMinOptions();
  assert(op_params != nullptr);

  auto operation = model_recipe->add_operation();

  operation->set_type("ArgMin");

  auto op_options = operation->mutable_argmin_options();

  op_options->set_output_type(as_tflchef_type(op_params->output_type()));

  return operation;
}

} // namespace tflchef
