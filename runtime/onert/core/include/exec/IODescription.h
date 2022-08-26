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

#ifndef __ONERT_EXEC_IO_DESCRIPTION_H__
#define __ONERT_EXEC_IO_DESCRIPTION_H__

#include <vector>
#include <unordered_map>
#include <semaphore.h>

#include "ir/OperandInfo.h"
#include "ir/Index.h"

namespace onert
{
namespace exec
{

struct InputDesc
{
  const ir::OperandInfo info;
  const void *buffer;
  const size_t size;
  const ir::Layout layout;

  InputDesc(void) = delete;
  InputDesc(const ir::OperandInfo &i, const void *b, const size_t s, ir::Layout l)
    : info(i), buffer(b), size(s), layout(l)
  {
  }

  InputDesc(const InputDesc &obj) = default;
  InputDesc &operator=(const InputDesc &) = delete;
  ~InputDesc() = default;
};

struct OutputDesc
{
  // not `const` because shape should be modified after execution in case when output is
  // a dynamic tensor
  ir::OperandInfo info;
  void *buffer;
  const size_t size;
  const ir::Layout layout;

  OutputDesc(void) = delete;
  OutputDesc(const ir::OperandInfo &info, void *buffer, const size_t size, ir::Layout layout)
    : info(info), buffer(buffer), size(size), layout(layout)
  {
  }

  OutputDesc(const OutputDesc &obj) = default;
  OutputDesc &operator=(const OutputDesc &) = delete;
  ~OutputDesc() = default;
};

struct IODescription
{
  std::vector<std::unique_ptr<InputDesc>> inputs;
  std::vector<std::unique_ptr<OutputDesc>> outputs;
  // Contains shape of input set by nnfw_set_input_tensorinfo(..)
  std::unordered_map<ir::IOIndex, ir::Shape> dynamic_input_shapes;

  IODescription(void) = default;

  IODescription(const IODescription &obj) : dynamic_input_shapes(obj.dynamic_input_shapes)
  {
    for (auto &input : obj.inputs)
    {
      inputs.emplace_back(
        std::make_unique<InputDesc>(input->info, input->buffer, input->size, input->layout));
    }
    for (auto &output : obj.outputs)
    {
      outputs.emplace_back(
        std::make_unique<OutputDesc>(output->info, output->buffer, output->size, output->layout));
    }
  }
  IODescription &operator=(const IODescription &rhs) = delete;
  ~IODescription() = default;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_IO_DESCRIPTION_H__
