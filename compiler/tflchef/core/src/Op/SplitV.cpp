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

#include "SplitV.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> SplitVChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_split_v_options());

  auto num_splits = operation.split_v_options().num_splits();

  tflite::SplitVOptionsBuilder split_v_options_builder{fbb};
  split_v_options_builder.add_num_splits(num_splits);

  return split_v_options_builder.Finish().Union();
}

std::unique_ptr<OpChef> SplitVChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new SplitVChef{operation}};
}
