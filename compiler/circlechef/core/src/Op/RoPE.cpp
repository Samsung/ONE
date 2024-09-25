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

#include "RoPE.h"

#include "Convert.h"

flatbuffers::Offset<void> RoPEChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);
  assert(operation.has_rope_options());

  circle::RoPEOptionsBuilder options_builder{fbb};
  options_builder.add_mode(static_cast<circle::RoPEMode>(operation.rope_options().mode()));

  return options_builder.Finish().Union();
}

std::unique_ptr<OpChef> RoPEChefFactory::create(const circlechef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new RoPEChef{operation}};
}
