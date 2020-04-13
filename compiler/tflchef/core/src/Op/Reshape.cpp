/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "Reshape.h"
#include "Convert.h"

#include <cassert>
#include <vector>

namespace
{

std::vector<int32_t> vector_new_shape(const tflchef::ReshapeOptions &options)
{
  std::vector<int32_t> shapes;

  for (int i = 0; i < options.new_shape_size(); ++i)
  {
    shapes.push_back(options.new_shape(i));
  }

  return shapes;
}

} // namespace

flatbuffers::Offset<void> ReshapeChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_reshape_options());

  auto options = operation.reshape_options();
  auto shapes = vector_new_shape(options);
  // Note: 'CreateVector' should be placed before 'options_builder'
  //       Read flatbuffers.h 'void NotNested()' for more information
  auto fb_new_shape = fbb.CreateVector(shapes);

  tflite::ReshapeOptionsBuilder options_builder{fbb};

  options_builder.add_new_shape(fb_new_shape);

  return options_builder.Finish().Union();
}

std::unique_ptr<OpChef> ReshapeChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new ReshapeChef{operation}};
}
