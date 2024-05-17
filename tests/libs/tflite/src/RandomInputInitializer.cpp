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
namespace
{

template <typename T>
void setValue(nnfw::misc::RandomGenerator &randgen, const TfLiteTensor *tensor)
{
  auto tensor_view = nnfw::tflite::TensorView<T>::make(tensor);

  nnfw::misc::tensor::iterate(tensor_view.shape())
    << [&](const nnfw::misc::tensor::Index &ind) { tensor_view.at(ind) = randgen.generate<T>(); };
}

} // namespace

void RandomInputInitializer::run(TfLiteInterpreter &interp)
{
  const auto input_count = TfLiteInterpreterGetInputTensorCount(&interp);
  for (int32_t idx = 0; idx < input_count; idx++)
  {
    auto tensor = TfLiteInterpreterGetInputTensor(&interp, idx);
    auto const tensor_type = TfLiteTensorType(tensor);
    switch (tensor_type)
    {
      case kTfLiteFloat32:
        setValue<float>(_randgen, tensor);
        break;
      case kTfLiteInt32:
        setValue<int32_t>(_randgen, tensor);
        break;
      case kTfLiteUInt8:
        setValue<uint8_t>(_randgen, tensor);
        break;
      case kTfLiteBool:
        setValue<bool>(_randgen, tensor);
        break;
      case kTfLiteInt8:
        setValue<int8_t>(_randgen, tensor);
        break;
      default:
        throw std::runtime_error{"Not supported input type"};
    }
  }
}

} // namespace tflite
} // namespace nnfw
