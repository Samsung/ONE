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

#include "SparseToDense.h"
#include "Convert.h"

#include <cassert>

flatbuffers::Offset<void> SparseToDenseChef::value(flatbuffers::FlatBufferBuilder &fbb) const
{
  auto &operation = (*_operation);

  assert(operation.has_sparse_to_dense_options());

  auto tflite_validate_indices = operation.sparse_to_dense_options().validate_indices();

  tflite::SparseToDenseOptionsBuilder sparse_to_dense_options_builder(fbb);
  sparse_to_dense_options_builder.add_validate_indices(tflite_validate_indices);

  return sparse_to_dense_options_builder.Finish().Union();
}

std::unique_ptr<OpChef> SparseToDenseChefFactory::create(const tflchef::Operation *operation) const
{
  return std::unique_ptr<OpChef>{new SparseToDenseChef{operation}};
}
