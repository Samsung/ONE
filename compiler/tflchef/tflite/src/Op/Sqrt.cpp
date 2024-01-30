/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Sqrt.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpSqrt::filler(const tflite::Operator *op, TFliteImport *import,
                          tflchef::ModelRecipe *model_recipe) const
{
  // Nothing to do with filler
  // But input has filler for constant inputs
  const auto &inputs = *op->inputs();

  const tflite::Tensor *tensor = import->tensors()->Get(inputs[0]);
  if (tensor->type() == tflite::TensorType::TensorType_FLOAT32)
  {
    const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());
    if (buffer && buffer->data())
    {
      auto vec = extract_buffer<float>(buffer);
      import->set_tensor_filler(inputs[0], vec);
    }
  }
}

tflchef::Operation *TFliteOpSqrt::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  operation->set_type("Sqrt");

  return operation;
}

} // namespace tflchef
