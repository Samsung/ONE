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

#include "Unique.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> UniqueChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_unique_options());

  auto tflite_out_idx = as_tflite_tensortype(operation.unique_options().idx_out_type());

  tflite::UniqueOptionsBuilder unique_options_builder{fbb};
  unique_options_builder.add_idx_out_type(tflite_out_idx);

  return unique_options_builder.Finish().Union();
}

std::unique_ptr<OpChef> UniqueChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new UniqueChef{operation}};
}
