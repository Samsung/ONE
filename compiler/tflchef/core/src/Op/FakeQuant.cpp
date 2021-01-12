/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "FakeQuant.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> FakeQuantChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);
  assert(operation.has_fakequant_options());

  auto options = operation.fakequant_options();

  tflite::FakeQuantOptionsBuilder fq_options_builder{fbb};
  fq_options_builder.add_min(options.min());
  fq_options_builder.add_max(options.max());
  fq_options_builder.add_num_bits(options.num_bits());
  fq_options_builder.add_narrow_range(options.narrow_range());

  return fq_options_builder.Finish().Union();
}

std::unique_ptr<OpChef> FakeQuantChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new FakeQuantChef{operation}};
}
