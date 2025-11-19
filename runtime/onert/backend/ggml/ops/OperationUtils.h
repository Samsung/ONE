/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_GGML_OPS_OPERATION_UTILS_H__
#define __ONERT_BACKEND_GGML_OPS_OPERATION_UTILS_H__

#include <cstdint>

namespace onert::backend::ggml::ops
{

inline int32_t getAxis(uint32_t rank, int32_t axis)
{
  auto ret = axis;

  if (axis < 0)
  {
    ret += rank;
  }

  return ret;
}

} // namespace onert::backend::ggml::ops

#endif // __ONERT_BACKEND_GGML_OPS_OPERATION_UTILS_H__
