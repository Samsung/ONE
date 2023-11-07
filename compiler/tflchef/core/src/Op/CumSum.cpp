/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "CumSum.h"

flatbuffers::Offset<void> CumSumChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_cumsum_options());

  auto exclusive = operation.cumsum_options().exclusive();
  auto reverse = operation.cumsum_options().reverse();

  tflite::CumsumOptionsBuilder cumsum_options_builder{fbb};
  cumsum_options_builder.add_exclusive(exclusive);
  cumsum_options_builder.add_reverse(reverse);

  return cumsum_options_builder.Finish().Union();
}

std::unique_ptr<OpChef> CumSumChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new CumSumChef{operation}};
}
