
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

#include "common.h"

bool tensorInfoEqual(const nnfw_tensorinfo &info1, const nnfw_tensorinfo &info2)
{
  if (info1.dtype != info2.dtype)
    return false;
  if (info1.rank != info2.rank)
    return false;
  for (int i = 0; i < info1.rank; i++)
    if (info1.dims[i] != info2.dims[i])
      return false;
  return true;
}

uint64_t tensorInfoNumElements(const nnfw_tensorinfo &ti)
{
  uint64_t n = 1;
  for (uint32_t i = 0; i < ti.rank; ++i)
  {
    n *= ti.dims[i];
  }
  return n;
}
