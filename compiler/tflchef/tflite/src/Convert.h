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

#ifndef __CONVERT_H__
#define __CONVERT_H__

#include <mio/tflite/schema_generated.h>

#include <tflchef.pb.h>

namespace tflchef
{

tflchef::TensorType as_tflchef_type(const tflite::TensorType type);
tflchef::Activation as_tflchef_activation(const tflite::ActivationFunctionType type);
tflchef::Padding as_tflchef_padding(const tflite::Padding padding);
tflchef::MirrorPadMode as_tflchef_mirrorpadmode(const tflite::MirrorPadMode mode);
tflchef::DimensionType as_tflchef_sparse_dim_type(const tflite::DimensionType type);
tflchef::SparseIndexVecType as_tflchef_sparse_idx_vec_type(const tflite::SparseIndexVector type);

/**
 * @brief extract buffer data to std::vector<DT>
 */
template <typename DT> std::vector<DT> extract_buffer(const tflite::Buffer *buffer)
{
  assert(buffer->data() != nullptr);
  auto buffer_length = buffer->data()->size();
  auto num_elements = buffer_length / sizeof(DT);
  std::vector<DT> result(num_elements);
  std::memcpy(result.data(), buffer->data()->data(), buffer_length);
  return result;
}

template <typename T> std::vector<T> as_index_vector(const flatbuffers::Vector<T> *flat_array)
{
  std::vector<T> ret(flat_array->size());
  for (uint32_t i = 0; i < flat_array->size(); i++)
  {
    ret[i] = flat_array->Get(i);
  }
  return ret;
}

} // namespace tflchef

#endif // __CONVERT_H__
