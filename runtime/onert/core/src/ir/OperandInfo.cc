/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ir/OperandInfo.h"

#include <cassert>

namespace onert
{
namespace ir
{

size_t OperandInfo::total_size() const
{
  const auto data_type = _typeInfo.type();
  try
  {
    return _shape.num_elements() * sizeOfDataType(data_type);
  }
  catch (const std::runtime_error &e)
  {
    if (data_type != DataType::QUANT_UINT4_SYMM_PER_CHUNK &&
        data_type != DataType::QUANT_INT8_SYMM_PER_CHUNK)
      throw e;

    // Assume last dim is multiple of chunk size (32)
    assert(_shape.dim(_shape.rank() - 1) % 32 == 0);
    const auto num_chunks = _shape.num_elements() / 32;
    const auto chunk_size = data_type == DataType::QUANT_UINT4_SYMM_PER_CHUNK
                              ? (sizeof(uint8_t) * 32 / 2 + sizeof(uint16_t))
                              : (sizeof(uint8_t) * 32 + sizeof(uint16_t));
    return num_chunks * chunk_size;
  }
}

} // namespace ir
} // namespace onert
