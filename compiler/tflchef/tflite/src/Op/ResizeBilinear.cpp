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

#include "ResizeBilinear.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpResizeBilinear::filler(const tflite::Operator *op, TFliteImport *import,
                                    tflchef::ModelRecipe *model_recipe) const
{
  // size buffer has filler
  const auto &inputs = *op->inputs();

  const tflite::Tensor *tensor = import->tensors()->Get(inputs[1]);
  assert(tensor->type() == tflite::TensorType::TensorType_INT32);
  const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());

  if (buffer && buffer->data())
  {
    auto vec = extract_buffer<int32_t>(buffer);
    import->set_tensor_filler(inputs[1], vec);
  }
}

tflchef::Operation *TFliteOpResizeBilinear::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_ResizeBilinearOptions();
  assert(op_params != nullptr);

  operation->set_type("ResizeBilinear");

  auto op_options = operation->mutable_resize_bilinear_options();

  op_options->set_align_corners(op_params->align_corners());
  op_options->set_half_pixel_centers(op_params->half_pixel_centers());

  return operation;
}

} // namespace tflchef
