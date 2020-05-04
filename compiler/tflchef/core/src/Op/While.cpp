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
 * See the License for the specwhileic language governing permissions and
 * limitations under the License.
 */

#include "While.h"

flatbuffers::Offset<void> WhileChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_while_options());

  tflite::WhileOptionsBuilder while_options_builder{fbb};
  while_options_builder.add_cond_subgraph_index(operation.while_options().cond_subgraph_index());
  while_options_builder.add_body_subgraph_index(operation.while_options().body_subgraph_index());

  return while_options_builder.Finish().Union();
}

std::unique_ptr<OpChef> WhileChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new WhileChef{operation}};
}
