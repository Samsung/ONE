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
#include "util/Exceptions.h"

namespace onert::ir
{

size_t OperandInfo::total_size() const
{
  const auto data_type = _typeInfo.type();
  try
  {
    return _shape.num_elements() * sizeOfDataType(data_type);
  }
  catch (const onert::UnsupportedDataTypeException &e)
  {
    // Calculate total size for ggml block quantization type on exception handling
    // because it is rare case and we should care about performance on non-block case.
    if (data_type != DataType::QUANT_GGML_Q4_0 && data_type != DataType::QUANT_GGML_Q8_0)
      throw e;

    if (_shape.dim(_shape.rank() - 1) % 32 != 0)
      throw std::runtime_error{
        "Block quantization requires the last dimension to be a multiple of 32"};

    const auto num_blocks = _shape.num_elements() / 32;
    const auto block_size = data_type == DataType::QUANT_GGML_Q4_0
                              ? (sizeof(uint8_t) * 32 / 2 + sizeof(uint16_t))
                              : (sizeof(uint8_t) * 32 + sizeof(uint16_t));
    return num_blocks * block_size;
  }
}

} // namespace onert::ir
