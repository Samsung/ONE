/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MIO_CIRCLE09_HELPER_H__
#define __MIO_CIRCLE09_HELPER_H__

#include <mio/circle/schema_generated.h>

#include <vector>

namespace mio
{
namespace circle
{

::circle::BuiltinOperator builtin_code_neutral(const ::circle::OperatorCode *opcode);
bool is_valid(const ::circle::OperatorCode *opcode);
bool is_custom(const ::circle::OperatorCode *opcode);
std::string opcode_name(const ::circle::OperatorCode *opcode);
const char *tensor_type(const ::circle::Tensor *tensor);
const char *tensor_name(const ::circle::Tensor *tensor);

template <typename T> std::vector<T> as_index_vector(const flatbuffers::Vector<T> *flat_array)
{
  if (flat_array == nullptr)
  {
    throw std::runtime_error("flat array is nullptr");
  }

  std::vector<T> ret(flat_array->size());
  for (uint32_t i = 0; i < flat_array->size(); i++)
  {
    ret[i] = flat_array->Get(i);
  }
  return ret;
}

} // namespace circle
} // namespace mio

#endif // __MIO_CIRCLE09_HELPER_H__
