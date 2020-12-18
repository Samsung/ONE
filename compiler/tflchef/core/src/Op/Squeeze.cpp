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

#include "Squeeze.h"
#include "Convert.h"

#include <cassert>
#include <vector>

flatbuffers::Offset<void> SqueezeChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_squeeze_options());

  const auto &options = operation.squeeze_options();
  // Note: 'CreateVector' should be placed before 'CreateOptions'
  //       Read flatbuffers.h 'void NotNested()' for more information
  auto fb_squeeze_dims =
    fbb.CreateVector(options.squeeze_dim().data(), options.squeeze_dim().size());

  return tflite::CreateSqueezeOptions(fbb, fb_squeeze_dims).Union();
}

std::unique_ptr<OpChef> SqueezeChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new SqueezeChef{operation}};
}
