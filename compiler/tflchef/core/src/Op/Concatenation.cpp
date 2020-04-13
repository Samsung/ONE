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

#include "Concatenation.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> ConcatenationChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_concatenation_options());

  auto options = operation.concatenation_options();

  auto tflite_activation = as_tflite_activation(options.activation());

  tflite::ConcatenationOptionsBuilder options_builder{fbb};

  options_builder.add_axis(options.axis());
  options_builder.add_fused_activation_function(tflite_activation);

  return options_builder.Finish().Union();
}

std::unique_ptr<OpChef> ConcatenationChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new ConcatenationChef{operation}};
}
