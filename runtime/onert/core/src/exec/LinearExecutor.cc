/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "LinearExecutor.h"

#include <compiler/CodeMap.h>

namespace onert
{
namespace exec
{

void LinearExecutor::executeImpl()
{
  _subject.notifyModelBegin(this);
  for (auto &&code : _code)
  {
    const auto op_seq = code->op_seq;
    const auto backend = code->lower_info->backend();
    _subject.notifyJobBegin(this, op_seq, backend);

    // run function sequence considering dynamic tensor
    if (backend->config()->supportDynamicTensor())
    {
      auto dyn_code = dynamic_cast<onert::compiler::CodeAndInfoForDynamicTensor *>(code.get());
      assert(dyn_code);

      code->fn_seq->run(dyn_code->op_seq, _lowered_graph->graph().operands(),
                        dyn_code->dynamic_tensor_manager, dyn_code->tensor_registry);
    }
    else
    {
      code->fn_seq->run();
    }

    _subject.notifyJobEnd(this, op_seq, backend);
  }
  _subject.notifyModelEnd(this);
}

} // namespace exec
} // namespace onert
