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

#include "FillerHelper.h"

#include "Convert.h"

namespace tflchef
{

void fill_tensor_to_import(int32_t idx, TFliteImport *import)
{
  const tflite::Tensor *tensor = import->tensors()->Get(idx);
  if (tensor != nullptr)
  {
    if (tensor->type() == tflite::TensorType::TensorType_INT32)
    {
      const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());
      if (buffer && buffer->data())
      {
        auto vec = extract_buffer<int32_t>(buffer);
        import->set_tensor_filler(idx, vec);
      }
    }
    else if (tensor->type() == tflite::TensorType::TensorType_FLOAT32)
    {
      const tflite::Buffer *buffer = import->buffers()->Get(tensor->buffer());
      if (buffer && buffer->data())
      {
        auto vec = extract_buffer<float>(buffer);
        import->set_tensor_filler(idx, vec);
      }
    }
  }
}

} // namespace tflchef

// helpers of common codes for filling inputs
namespace tflchef
{

void fill_two_inputs(const tflite::Operator *op, TFliteImport *import)
{
  const std::vector<int32_t> &inputs = as_index_vector(op->inputs());
  assert(inputs.size() == 2);

  fill_tensor_to_import(inputs[0], import);
  fill_tensor_to_import(inputs[1], import);
}

} // namespace tflchef
