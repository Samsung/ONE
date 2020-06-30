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

#include "BCQGather.h"

#include "Convert.h"

flatbuffers::Offset<void> BCQGatherChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_bcq_gather_options());

  circle::BCQGatherOptionsBuilder bcq_gather_options_builder{fbb};
  bcq_gather_options_builder.add_input_hidden_size(
      operation.bcq_gather_options().input_hidden_size());
  bcq_gather_options_builder.add_axis(operation.bcq_gather_options().axis());

  return bcq_gather_options_builder.Finish().Union();
}

std::unique_ptr<OpChef> BCQGatherChefFactory::create(const circlechef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new BCQGatherChef{operation}};
}
