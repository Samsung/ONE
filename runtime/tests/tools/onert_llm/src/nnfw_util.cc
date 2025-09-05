/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "nnfw_util.h"

#include <cassert>
#include <iostream>
#include <string>

namespace onert_llm
{
uint64_t num_elems(const nnfw_tensorinfo *ti)
{
  uint64_t n = 1;
  for (int32_t i = 0; i < ti->rank; ++i)
  {
    assert(ti->dims[i] >= 0);
    n *= ti->dims[i];
  }
  return n;
}

uint64_t bufsize_for(const nnfw_tensorinfo *ti)
{
  static uint32_t elmsize[] = {
    sizeof(float),   /* NNFW_TYPE_TENSOR_FLOAT32 */
    sizeof(int),     /* NNFW_TYPE_TENSOR_INT32 */
    sizeof(uint8_t), /* NNFW_TYPE_TENSOR_QUANT8_ASYMM */
    sizeof(bool),    /* NNFW_TYPE_TENSOR_BOOL = 3 */
    sizeof(uint8_t), /* NNFW_TYPE_TENSOR_UINT8 = 4 */
    sizeof(int64_t), /* NNFW_TYPE_TENSOR_INT64 = 5 */
    sizeof(int8_t),  /* NNFW_TYPE_TENSOR_QUANT8_ASYMM_SIGNED = 6 */
    sizeof(int16_t), /* NNFW_TYPE_TENSOR_QUANT16_SYMM_SIGNED = 7 */
  };
  return elmsize[ti->dtype] * num_elems(ti);
}

uint64_t has_dynamic_dim(const nnfw_tensorinfo *ti)
{
  for (int32_t i = 0; i < ti->rank; ++i)
  {
    if (ti->dims[i] < 0)
      return true;
  }
  return false;
}

void print_version()
{
  uint32_t version;
  NNPR_ENSURE_STATUS(nnfw_query_info_u32(NULL, NNFW_INFO_ID_VERSION, &version));
  std::cout << "onert_run (nnfw runtime: v" << (version >> 24) << "."
            << ((version & 0x0000FF00) >> 8) << "." << (version & 0xFF) << ")" << std::endl;
}

} // namespace onert_llm
