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

#include "NonMaxSuppressionV5.h"

#include "Convert.h"
#include "FillerHelper.h"

namespace tflchef
{

void TFliteOpNonMaxSuppressionV5::filler(const tflite::Operator *op, TFliteImport *import,
                                         tflchef::ModelRecipe *model_recipe) const
{
  const auto &inputs = *op->inputs();

  const tflite::Tensor *max_output_size_tensor = import->tensors()->Get(inputs[2]);
  assert(max_output_size_tensor->type() == tflite::TensorType::TensorType_INT32);

  const tflite::Tensor *iou_threshold_tensor = import->tensors()->Get(inputs[3]);
  assert(iou_threshold_tensor->type() == tflite::TensorType::TensorType_FLOAT32);

  const tflite::Tensor *score_threshold_tensor = import->tensors()->Get(inputs[4]);
  assert(score_threshold_tensor->type() == tflite::TensorType::TensorType_FLOAT32);

  const tflite::Tensor *soft_nms_sigma_tensor = import->tensors()->Get(inputs[5]);
  assert(soft_nms_sigma_tensor->type() == tflite::TensorType::TensorType_FLOAT32);

  for (int32_t index = 2; index < 6; ++index)
  {
    fill_tensor_to_import(inputs[index], import);
  }
}

tflchef::Operation *TFliteOpNonMaxSuppressionV5::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  operation->set_type("NonMaxSuppressionV5");

  return operation;
}

} // namespace tflchef
