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

#include "NonMaxSuppressionV4.h"

flatbuffers::Offset<void> NonMaxSuppressionV4Chef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  tflite::NonMaxSuppressionV4OptionsBuilder options_builder{fbb};

  return options_builder.Finish().Union();
}

std::unique_ptr<OpChef>
NonMaxSuppressionV4ChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new NonMaxSuppressionV4Chef{operation}};
}
