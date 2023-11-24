/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

#ifndef LUCI_INTERPRETER_PAL_FULLY_CONNECTED_H
#define LUCI_INTERPRETER_PAL_FULLY_CONNECTED_H

#include "PALFullyConnectedCommon.h"

namespace luci_interpreter_pal
{

template <>
inline void FullyConnected(const luci_interpreter_pal::FullyConnectedParams &params,
                           const int32_t *input_shape, const int8_t *input_data,
                           const int32_t *filter_shape, const int8_t *filter_data,
                           const int32_t *bias_data, const int32_t *output_shape,
                           int8_t *output_data, uint32_t, uint32_t)
{
  // MARK: At this moment this operation doesn't support
  assert(false && "FullyConnected INT8 NYI");
  (void)params;
  (void)input_shape;
  (void)input_data;
  (void)filter_shape;
  (void)filter_data;
  (void)bias_data;
  (void)output_shape;
  (void)output_data;
}

template <>
inline void FullyConnected(const luci_interpreter_pal::FullyConnectedParams &, const int32_t *,
                           const int16_t *, const int32_t *, const int8_t *, const int64_t *,
                           const int32_t *, int16_t *, uint32_t, uint32_t)
{
  // MARK: At this moment this operation doesn't support
  assert(false && "FullyConnected INT16 NYI");
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_FULLY_CONNECTED_H
