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

#include "StridedSlice.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> StridedSliceChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_strided_slice_options());

  tflite::StridedSliceOptionsBuilder strided_slice_options_builder{fbb};
  strided_slice_options_builder.add_begin_mask(operation.strided_slice_options().begin_mask());
  strided_slice_options_builder.add_end_mask(operation.strided_slice_options().end_mask());
  strided_slice_options_builder.add_ellipsis_mask(
    operation.strided_slice_options().ellipsis_mask());
  strided_slice_options_builder.add_new_axis_mask(
    operation.strided_slice_options().new_axis_mask());
  strided_slice_options_builder.add_shrink_axis_mask(
    operation.strided_slice_options().shrink_axis_mask());

  return strided_slice_options_builder.Finish().Union();
}

std::unique_ptr<OpChef> StridedSliceChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new StridedSliceChef{operation}};
}
