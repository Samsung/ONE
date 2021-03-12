/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tflite/RandomInputInitializer.h"
#include "tflite/TensorView.h"

#include <misc/tensor/IndexIterator.h>

namespace nnfw
{
namespace tflite
{

void RandomInputInitializer::run(::tflite::Interpreter &interp)
{
  for (const auto &tensor_idx : interp.inputs())
  {
    TfLiteTensor *tensor = interp.tensor(tensor_idx);
    switch (tensor->type)
    {
      case kTfLiteFloat32:
        setValue<float>(interp, tensor_idx);
        break;
      case kTfLiteInt32:
        setValue<int32_t>(interp, tensor_idx);
        break;
      case kTfLiteUInt8:
        setValue<uint8_t>(interp, tensor_idx);
        break;
      case kTfLiteBool:
        setValue<bool>(interp, tensor_idx);
        break;
      case kTfLiteInt8:
        setValue<int8_t>(interp, tensor_idx);
        break;
      default:
        throw std::runtime_error{"Not supported input type"};
    }
  }
}

template <typename T>
void RandomInputInitializer::setValue(::tflite::Interpreter &interp, int tensor_idx)
{
  auto tensor_view = nnfw::tflite::TensorView<T>::make(interp, tensor_idx);

  nnfw::misc::tensor::iterate(tensor_view.shape())
    << [&](const nnfw::misc::tensor::Index &ind) { tensor_view.at(ind) = _randgen.generate<T>(); };
}

} // namespace tflite
} // namespace nnfw
