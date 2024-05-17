/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_EXEC_EXECUTION_CONTEXT_H__
#define __ONERT_EXEC_EXECUTION_CONTEXT_H__

#include <vector>
#include <memory>

#include "ir/OperandInfo.h"
#include "ir/Index.h"

namespace onert
{
namespace exec
{

struct InputDesc
{
  ir::OperandInfo info;
  const void *buffer;
  size_t size;
  ir::Layout layout;

  InputDesc(void) = delete;
  InputDesc(const ir::OperandInfo &info)
    : info(info), buffer(nullptr), size(0), layout(ir::Layout::NHWC)
  {
  }
};

struct OutputDesc
{
  ir::OperandInfo info;
  void *buffer;
  size_t size;
  ir::Layout layout;

  OutputDesc(void) = delete;
  OutputDesc(const ir::OperandInfo &info)
    : info(info), buffer(nullptr), size(0), layout(ir::Layout::NHWC)
  {
  }
};

struct IODescription
{
  std::vector<std::unique_ptr<InputDesc>> inputs;
  std::vector<std::unique_ptr<OutputDesc>> outputs;
};

struct ExecutionOptions
{
  bool dump_minmax = false;
  bool trace = false;
  bool profile = false;

  static std::unique_ptr<ExecutionOptions> fromGlobalConfig();
};

struct ExecutionContext
{
  IODescription desc;
  bool shape_updated = false; // Require shape inference and buffer size calculation
  ExecutionOptions options;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTION_CONTEXT_H__
