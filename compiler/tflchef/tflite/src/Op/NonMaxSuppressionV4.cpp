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

#include "NonMaxSuppressionV4.h"

#include "Convert.h"

namespace tflchef
{

void TFliteOpNonMaxSuppressionV4::filler(const tflite::Operator *op, TFliteImport *import,
                                         tflchef::ModelRecipe *model_recipe) const
{
  const auto &inputs = *op->inputs();

  // for input max_output_size
  const tflite::Tensor *tensor = import->tensors()->Get(inputs[2]);
  assert(tensor->type() == tflite::TensorType::TensorType_INT32);
  const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());
  if (buffer && buffer->data() != nullptr)
  {
    auto vec = extract_buffer<int32_t>(buffer);
    import->set_tensor_filler(inputs[2], vec);
  }

  // for iou and score thresholds
  for (int32_t index = 3; index <= 4; ++index)
  {
    const tflite::Tensor *tensor = import->tensors()->Get(inputs[index]);
    assert(tensor->type() == tflite::TensorType::TensorType_FLOAT32);
    const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());
    if (buffer && buffer->data() != nullptr)
    {
      auto vec = extract_buffer<float>(buffer);
      import->set_tensor_filler(inputs[index], vec);
    }
  }
}

tflchef::Operation *TFliteOpNonMaxSuppressionV4::build(const tflite::Operator *op,
                                                       TFliteImport *import,
                                                       tflchef::ModelRecipe *model_recipe) const
{
  auto operation = model_recipe->add_operation();

  operation->set_type("NonMaxSuppressionV4");

  return operation;
}

} // namespace tflchef
