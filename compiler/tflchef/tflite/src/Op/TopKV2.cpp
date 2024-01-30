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

#include "TopKV2.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpTopKV2::filler(const tflite::Operator *op, TFliteImport *import,
                            tflchef::ModelRecipe *model_recipe) const
{
  // filler for second input
  const auto &inputs = *op->inputs();

  const tflite::Tensor *tensor = import->tensors()->Get(inputs[1]);
  assert(tensor->type() == tflite::TensorType::TensorType_INT32 ||
         tensor->type() == tflite::TensorType::TensorType_INT64);
  const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());
  auto vec = extract_buffer<int32_t>(buffer);
  import->set_tensor_filler(inputs[1], vec);
}

tflchef::Operation *TFliteOpTopKV2::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  operation->set_type("TopKV2");

  // NOTE there is `sorted` attribute in TF but it's always true for TFlite

  return operation;
}

} // namespace tflchef
