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

#include "exec/FunctionSequence.h"

#include "ir/Operation.h"
#include "backend/IDynamicTensorManager.h"
#include "backend/ITensorRegistry.h"

namespace onert
{
namespace exec
{

// for OpSequence only with static tensor
void FunctionSequence::run()
{
  for (const auto &function : _functions)
  {
    function->run();
  }
}

void FunctionSequence::prepare()
{
  for (const auto &function : _functions)
  {
    function->prepare();
  }
}

void FunctionSequence::append(std::unique_ptr<IFunction> &&function)
{
  _functions.push_back(std::move(function));
}

void FunctionSequence::iterate(const std::function<void(IFunction &)> &fn)
{
  for (const auto &func : _functions)
  {
    fn(*func);
  }
}

void FunctionSequenceForDynamicBackend::run()
{
  if (_op_seq.size() != _functions.size())
    throw std::runtime_error("operation and functions should be mapped one by one");

  auto op_seq_iter = _op_seq.begin();
  for (const auto &function : _functions)
  {
    // set shape of output and allocate memory when needed
    auto &op = _operations_ctx.at(*op_seq_iter);
    op.accept(*_dyn_shape_inferer);

    // run kernel
    function->run();

    // deallocate input tensors which is no longer used
    _dyn_tensor_manager->deallocInput(*op_seq_iter);

    op_seq_iter++;
  }
}

} // namespace exec
} // namespace onert
