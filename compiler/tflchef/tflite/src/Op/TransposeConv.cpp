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

#include "TransposeConv.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpTransposeConv::filler(const tflite::Operator *op, TFliteImport *import,
                                   tflchef::ModelRecipe *model_recipe) const
{
  const auto &inputs = *op->inputs();

  const tflite::Tensor *tensor = import->tensors()->Get(inputs[0]);
  assert(tensor->type() == tflite::TensorType::TensorType_INT32);
  const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());

  if (buffer && buffer->data())
  {
    auto vec = extract_buffer<int32_t>(buffer);
    import->set_tensor_filler(inputs[0], vec);
  }

  // filter
  const tflite::Tensor *filter_tensor = import->tensors()->Get(inputs[1]);
  import->set_tensor_filler(inputs[1]);
}

tflchef::Operation *TFliteOpTransposeConv::build(const tflite::Operator *op, TFliteImport *import,
                                                 tflchef::ModelRecipe *model_recipe) const
{
  auto op_params = op->builtin_options_as_TransposeConvOptions();

  assert(op_params != nullptr);

  auto operation = model_recipe->add_operation();

  operation->set_type("TransposeConv");

  auto op_options = operation->mutable_transpose_conv_options();
  auto tflchef_activation = as_tflchef_activation(op_params->fused_activation_function());

  op_options->set_stride_h(op_params->stride_h());
  op_options->set_stride_w(op_params->stride_w());
  op_options->set_padding(as_tflchef_padding(op_params->padding()));
  op_options->set_activation(tflchef_activation);

  return operation;
}

} // namespace tflchef
