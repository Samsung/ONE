/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_MODEL_OPERAND_CONSTRAINT_H__
#define __ONERT_MODEL_OPERAND_CONSTRAINT_H__

#include <stdint.h>
#include <limits>
#include <set>

namespace onert::ir
{

class OperandConstraint
{
private:
  static inline const uint32_t INF = std::numeric_limits<uint32_t>::max();

public:
  static OperandConstraint createAny() { return OperandConstraint{0u, INF}; }
  static OperandConstraint createExact(uint32_t exact) { return OperandConstraint{exact, exact}; }
  static OperandConstraint createAtMost(uint32_t end) { return OperandConstraint{0u, end}; }
  static OperandConstraint createAtLeast(uint32_t begin) { return OperandConstraint{begin, INF}; }
  static OperandConstraint createInRange(uint32_t begin, uint32_t end)
  {
    return OperandConstraint{begin, end};
  }

private:
  OperandConstraint(uint32_t begin, uint32_t end) : _begin{begin}, _end{end} {}

public:
  bool check(uint32_t ind) const { return _begin <= ind && ind <= _end; }

private:
  uint32_t _begin;
  uint32_t _end;
};

} // namespace onert::ir

#endif // __ONERT_MODEL_OPERAND_CONSTRAINT_H__
