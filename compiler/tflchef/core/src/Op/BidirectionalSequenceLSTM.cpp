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

#include "BidirectionalSequenceLSTM.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void>
BidirectionalSequenceLSTMChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_bidirectional_sequence_lstm_options());

  tflite::BidirectionalSequenceLSTMOptionsBuilder options_builder(fbb);
  options_builder.add_fused_activation_function(
      as_tflite_activation(operation.bidirectional_sequence_lstm_options().activation()));
  options_builder.add_cell_clip(operation.bidirectional_sequence_lstm_options().cell_clip());
  options_builder.add_proj_clip(operation.bidirectional_sequence_lstm_options().proj_clip());
  options_builder.add_time_major(operation.bidirectional_sequence_lstm_options().time_major());
  options_builder.add_asymmetric_quantize_inputs(
      operation.bidirectional_sequence_lstm_options().asymmetric_quantize_inputs());
  options_builder.add_merge_outputs(
      operation.bidirectional_sequence_lstm_options().merge_outputs());

  return options_builder.Finish().Union();
}

std::unique_ptr<OpChef>
BidirectionalSequenceLSTMChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new BidirectionalSequenceLSTMChef{operation}};
}
