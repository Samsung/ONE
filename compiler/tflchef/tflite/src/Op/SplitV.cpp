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

#include "SplitV.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpSplitV::filler(const tflite::Operator *op, TFliteImport *import,
                            tflchef::ModelRecipe *model_recipe) const
{
  const auto &inputs = *op->inputs();

  // for input "size_splits" and "split_dim"
  for (int32_t idx = 1; idx <= 2; idx++)
  {
    const tflite::Tensor *tensor = import->tensors()->Get(inputs[idx]);
    assert(tensor->type() == tflite::TensorType::TensorType_INT32);
    const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());
    auto vec = extract_buffer<int32_t>(buffer);
    import->set_tensor_filler(inputs[idx], vec);
  }
}

tflchef::Operation *TFliteOpSplitV::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  operation->set_type("SplitV");

  auto op_options = operation->mutable_split_v_options();

  auto op_params = op->builtin_options_as_SplitVOptions();
  assert(op_params != nullptr);

  op_options->set_num_splits(op_params->num_splits());
  return operation;
}

} // namespace tflchef
