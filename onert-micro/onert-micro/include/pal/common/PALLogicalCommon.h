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

#ifndef ONERT_MICRO_EXECUTE_PAL_LOGICAL_COMMON_H
#define ONERT_MICRO_EXECUTE_PAL_LOGICAL_COMMON_H

#include "OMStatus.h"

namespace onert_micro::execute::pal
{

struct LogicalAndFn
{
  bool operator()(bool lhs, bool rhs) { return lhs && rhs; }
};

struct LogicalOrFn
{
  bool operator()(bool lhs, bool rhs) { return lhs || rhs; }
};

// ------------------------------------------------------------------------------------------------

template <class Fn>
OMStatus LogicalCommon(const int flat_size, const bool *input1_data, const bool *input2_data,
                       bool *output_data)
{
  Fn func;

  for (int i = 0; i < flat_size; ++i)
  {
    output_data[i] = func(input1_data[i], input2_data[i]);
  }

  return Ok;
}

} // namespace onert_micro::execute::pal

#endif // ONERT_MICRO_EXECUTE_PAL_LOGICAL_COMMON_H
