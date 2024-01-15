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

#ifndef __CONVERT_H__
#define __CONVERT_H__

#include <mio/circle/schema_generated.h>

#include <circlechef.pb.h>

namespace circlechef
{

circlechef::TensorType as_circlechef_type(const circle::TensorType type);
circlechef::Activation as_circlechef_activation(const circle::ActivationFunctionType type);
circlechef::Padding as_circlechef_padding(const circle::Padding padding);

/**
 * @brief extract buffer data to std::vector<DT>
 */
template <typename DT> std::vector<DT> extract_buffer(const circle::Buffer *buffer)
{
  auto buffer_length = buffer->data()->size();
  auto num_elements = buffer_length / sizeof(DT);
  std::vector<DT> result(num_elements);
  std::memcpy(result.data(), buffer->data()->data(), buffer_length);
  return result;
}

template <typename T> std::vector<T> as_index_vector(const flatbuffers::Vector<T> *flat_array)
{
  if (flat_array == nullptr)
    throw std::runtime_error("flat_array is nullptr");

  std::vector<T> ret(flat_array->size());
  for (uint32_t i = 0; i < flat_array->size(); i++)
  {
    ret[i] = flat_array->Get(i);
  }
  return ret;
}

} // namespace circlechef

#endif // __CONVERT_H__
