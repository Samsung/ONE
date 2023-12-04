/* Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ConvertTypes.h"

#include <cassert>
#include <stdexcept>

namespace luci
{
namespace compute
{

tflite::RuntimeShape tflite_shape(const loco::TensorShape &shape)
{
  tflite::RuntimeShape runtime_shape(shape.rank());
  for (uint32_t i = 0; i < shape.rank(); ++i)
  {
    if (not shape.dim(i).known())
      throw std::runtime_error("luci-comp tflite_shape shape unknown.");
    runtime_shape.SetDim(i, shape.dim(i).value());
  }
  return runtime_shape;
}

tflite::PaddingType tflite_padding(const PaddingType type)
{
  switch (type)
  {
    case PaddingType::kSame:
      return tflite::PaddingType::kSame;
    case PaddingType::kValid:
      return tflite::PaddingType::kValid;
    default:
      break;
  }
  throw std::runtime_error("luci-comp tflite_padding unsupported type.");
}

tflite::FullyConnectedWeightsFormat tflite_weights_format(const FullyConnectedWeightsFormat type)
{
  switch (type)
  {
    case FullyConnectedWeightsFormat::kDefault:
      return tflite::FullyConnectedWeightsFormat::kDefault;
    case FullyConnectedWeightsFormat::kShuffled4x16Int8:
      return tflite::FullyConnectedWeightsFormat::kShuffled4x16Int8;
    default:
      break;
  }
  throw std::runtime_error("luci-comp tflite_weights_format unsupported type.");
}

} // namespace compute
} // namespace luci
