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

#ifndef __ONERT_COMPILER_EXECUTION_BUILDER_H__
#define __ONERT_COMPILER_EXECUTION_BUILDER_H__

#include <memory>

#include "ir/OpSequence.h"
#include "exec/FunctionSequence.h"
#include "CodeMap.h"

namespace onert
{
namespace compiler
{

class ExecutionBuilder
{
public:
  void append(const ir::OpSequenceIndex index, CodeAndInfo &&code_and_info)
  {
    _code_map.emplace(index, std::move(code_and_info));
  }

  CodeMap releaseCodeMap() { return std::move(_code_map); }

private:
  CodeMap _code_map;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_EXECUTION_BUILDER_H__
