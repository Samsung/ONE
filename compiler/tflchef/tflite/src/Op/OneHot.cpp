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

#include "OneHot.h"
#include "Convert.h"

namespace tflchef
{

void TFliteOpOneHot::filler(const tflite::Operator *op, TFliteImport *import,
                            tflchef::ModelRecipe *model_recipe) const
{
  // only depth(second input) has constant on recipe cause depth value is used in shape inference.
  const auto &inputs = *op->inputs();

  const tflite::Tensor *tensor = import->tensors()->Get(inputs[1]);
  assert(tensor->type() == tflite::TensorType::TensorType_INT32);
  const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());

  if (buffer && buffer->data())
  {
    auto vec = extract_buffer<int32_t>(buffer);
    import->set_tensor_filler(inputs[1], vec);
  }

  // on/off can be dtype of input/output. let's support INT32/FLOAT32 for now
  for (int32_t index = 2; index <= 3; ++index)
  {
    const tflite::Tensor *tensor = import->tensors()->Get(inputs[index]);
    const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());
    if (buffer && buffer->data())
    {
      switch (tensor->type())
      {
        case tflite::TensorType::TensorType_INT32:
        {
          auto vec = extract_buffer<int32_t>(buffer);
          import->set_tensor_filler(inputs[index], vec);
          break;
        }

        case tflite::TensorType::TensorType_FLOAT32:
        {
          auto vec = extract_buffer<float>(buffer);
          import->set_tensor_filler(inputs[index], vec);
          break;
        }

        default:
          assert(false);
          break;
      }
    }
  }
}

tflchef::Operation *TFliteOpOneHot::build(RecipeChefContext *ctx) const
{
  tflchef::Operation *operation = ctx->chefop;
  const tflite::Operator *op = ctx->tflop;

  auto op_params = op->builtin_options_as_OneHotOptions();
  assert(op_params != nullptr);

  operation->set_type("OneHot");

  auto op_options = operation->mutable_onehot_options();

  op_options->set_axis(op_params->axis());

  return operation;
}

} // namespace tflchef
