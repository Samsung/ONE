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

#include "FullyConnected.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> FullyConnectedChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_fullyconnected_options());

  auto circle_activation = as_circle_activation(operation.fullyconnected_options().activation());

  circle::FullyConnectedOptionsBuilder fc_options_builder{fbb};
  fc_options_builder.add_fused_activation_function(circle_activation);
  fc_options_builder.add_keep_num_dims(operation.fullyconnected_options().keep_num_dims());

  return fc_options_builder.Finish().Union();
}

std::unique_ptr<OpChef>
FullyConnectedChefFactory::create(const circlechef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new FullyConnectedChef{operation}};
}
