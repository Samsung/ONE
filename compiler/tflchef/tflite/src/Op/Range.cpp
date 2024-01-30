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

#include "Range.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpRange::filler(const tflite::Operator *op, TFliteImport *import,
                           tflchef::ModelRecipe *model_recipe) const
{
  // filler for all inputs
  const auto &inputs = *op->inputs();

  for (int index = 0; index < 3; ++index)
  {
    const tflite::Tensor *tensor = import->tensors()->Get(inputs[index]);
    const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());
    if (tensor->type() == tflite::TensorType::TensorType_INT32)
    {
      auto vec = extract_buffer<int32_t>(buffer);
      import->set_tensor_filler(inputs[index], vec);
    }
    else if (tensor->type() == tflite::TensorType::TensorType_FLOAT32)
    {
      auto vec = extract_buffer<float>(buffer);
      import->set_tensor_filler(inputs[index], vec);
    }
    else
    {
      assert(false && "Invalid tensor type");
    }
  }
}

tflchef::Operation *TFliteOpRange::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  operation->set_type("Range");

  return operation;
}

} // namespace tflchef
