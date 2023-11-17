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

#ifndef LUCI_INTERPRETER_PAL_BROADCASTTO_H
#define LUCI_INTERPRETER_PAL_BROADCASTTO_H

#include <tensorflow/lite/kernels/internal/reference/broadcast_to.h>

namespace luci_interpreter_pal
{

static inline void BroadcastTo(const tflite::RuntimeShape &input_shape, const char *input_data,
                               const tflite::RuntimeShape &output_shape, char *output_data,
                               TfLiteType data_type)
{
  // BroadcastTo op supports up to 8 kMaxDims in tensorflow.
  // but, currently we support up to 5 dims because there is a compiler bug in 7.4.0 gcc version.
  // https://github.com/tensorflow/tensorflow/blob/932af96ae91b4fa647dc50ad0f14c3e0b60affab/tensorflow/lite/kernels/broadcast_to.cc#L118
  constexpr int kMaxDims = 5;
  tflite::reference_ops::BroadcastTo<kMaxDims>(input_shape, input_data, output_shape, output_data,
                                               data_type);
}

} // namespace luci_interpreter_pal

#endif // LUCI_INTERPRETER_PAL_BROADCASTTO_H
