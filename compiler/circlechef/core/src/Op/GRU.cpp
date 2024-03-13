/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "GRU.h"

#include "Convert.h"

flatbuffers::Offset<void> GRUChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_gru_options());
  auto circle_activation = as_circle_activation(operation.gru_options().activation());
  auto return_sequences = operation.gru_options().return_sequences();
  auto time_major = operation.gru_options().time_major();

  circle::GRUOptionsBuilder options_builder{fbb};
  options_builder.add_fused_activation_function(circle_activation);
  options_builder.add_return_sequences(return_sequences);
  options_builder.add_time_major(time_major);

  return options_builder.Finish().Union();
}

std::unique_ptr<OpChef> GRUChefFactory::create(const circlechef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new GRUChef{operation}};
}
