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

#include "ArgMin.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> ArgMinChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_argmin_options());

  auto tflite_output_type = as_tflite_tensortype(operation.argmin_options().output_type());

  tflite::ArgMinOptionsBuilder argmin_options_builder{fbb};
  argmin_options_builder.add_output_type(tflite_output_type);

  return argmin_options_builder.Finish().Union();
}

std::unique_ptr<OpChef> ArgMinChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new ArgMinChef{operation}};
}
