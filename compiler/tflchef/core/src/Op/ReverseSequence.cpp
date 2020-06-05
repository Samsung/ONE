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

#include "ReverseSequence.h"

flatbuffers::Offset<void> ReverseSequenceChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_reverse_sequence_options());

  auto options = operation.reverse_sequence_options();

  auto tflite_seq_dim = options.seq_dim();
  auto tflite_batch_dim = options.batch_dim();

  tflite::ReverseSequenceOptionsBuilder options_builder{fbb};

  options_builder.add_seq_dim(tflite_seq_dim);
  options_builder.add_batch_dim(tflite_batch_dim);

  return options_builder.Finish().Union();
}

std::unique_ptr<OpChef>
ReverseSequenceChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new ReverseSequenceChef{operation}};
}
