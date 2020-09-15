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

#include "UnidirectionalSequenceLSTM.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void>
UnidirectionalSequenceLSTMChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_unidirectional_sequence_lstm_options());

  tflite::UnidirectionalSequenceLSTMOptionsBuilder options_builder(fbb);
  options_builder.add_fused_activation_function(
      as_tflite_activation(operation.unidirectional_sequence_lstm_options().activation()));
  options_builder.add_cell_clip(operation.unidirectional_sequence_lstm_options().cell_clip());
  options_builder.add_proj_clip(operation.unidirectional_sequence_lstm_options().proj_clip());
  options_builder.add_time_major(operation.unidirectional_sequence_lstm_options().time_major());

  return options_builder.Finish().Union();
}

std::unique_ptr<OpChef>
UnidirectionalSequenceLSTMChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new UnidirectionalSequenceLSTMChef{operation}};
}
