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

#include "StridedSlice.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpStridedSlice::filler(const tflite::Operator *op, TFliteImport *import,
                                  tflchef::ModelRecipe *model_recipe) const
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());

  // for begin, end and strides
  for (int32_t index = 1; index <= 3; ++index)
  {
    const tflite::Tensor *tensor = import->tensors()->Get(inputs[index]);
    assert(tensor->type() == tflite::TensorType::TensorType_INT32);
    const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());
    auto vec = extract_buffer<int32_t>(buffer);
    import->set_tensor_filler(inputs[index], vec);
  }
}

tflchef::Operation *TFliteOpStridedSlice::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_StridedSliceOptions();
  assert(op_params != nullptr);

  operation->set_type("StridedSlice");

  auto op_options = operation->mutable_strided_slice_options();

  op_options->set_begin_mask(op_params->begin_mask());
  op_options->set_end_mask(op_params->end_mask());
  op_options->set_ellipsis_mask(op_params->ellipsis_mask());
  op_options->set_new_axis_mask(op_params->new_axis_mask());
  op_options->set_shrink_axis_mask(op_params->shrink_axis_mask());
  return operation;
}

} // namespace tflchef
