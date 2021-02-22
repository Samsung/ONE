/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "tflite/FeatureView.h"
#include "tflite/TensorUtils.h"

#include <cassert>

namespace nnfw
{
namespace tflite
{

nnfw::misc::feature::Shape getFeatureShape(const TfLiteTensor *tensor)
{
  nnfw::misc::feature::Shape shape{tensor->dims->data[3], tensor->dims->data[1],
                                   tensor->dims->data[2]};

  return shape;
}

FeatureView<float>::FeatureView(::tflite::Interpreter &interp, const InputIndex &index)
{
  const auto tensor_index = interp.inputs().at(index.asInt());
  auto tensor_ptr = interp.tensor(tensor_index);

  assert(isFloatTensor(tensor_ptr));
  assert(isFeatureTensor(tensor_ptr));

  _shape = getFeatureShape(tensor_ptr);
  _base = interp.typed_tensor<float>(tensor_index);
}

FeatureView<float>::FeatureView(::tflite::Interpreter &interp, const OutputIndex &index)
{
  const auto tensor_index = interp.outputs().at(index.asInt());
  auto tensor_ptr = interp.tensor(tensor_index);

  assert(isFloatTensor(tensor_ptr));
  assert(isFeatureTensor(tensor_ptr));

  _shape = getFeatureShape(tensor_ptr);
  _base = interp.typed_tensor<float>(tensor_index);
}

float FeatureView<float>::at(uint32_t ch, uint32_t row, uint32_t col) const
{
  return *(_base + getElementOffset(ch, row, col));
}

float &FeatureView<float>::at(uint32_t ch, uint32_t row, uint32_t col)
{
  return *(_base + getElementOffset(ch, row, col));
}

} // namespace tflite
} // namespace nnfw
