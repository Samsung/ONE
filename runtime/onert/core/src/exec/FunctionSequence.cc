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
#include "util/logging.h"

namespace onert
{
namespace exec
{

void FunctionSequence::run()
{
  // TODO Find out when `_enable_dynamic_shape_inferer` is true but `_dynamic_tensor_ctx` is false
  if (_enable_dynamic_shape_inferer && _dynamic_tensor_ctx)
  {
    if (_dynamic_tensor_ctx->op_seq->size() != _functions.size())
      throw std::runtime_error("operation and functions should be mapped one by one");

    auto op_seq_iter = _dynamic_tensor_ctx->op_seq->begin();
    for (const auto &function : _functions)
    {
      // set shape of output and allocate memory when needed
      auto &op = _dynamic_tensor_ctx->operations->at(*op_seq_iter);
      op.accept(*_dynamic_tensor_ctx->dynamic_shape_inferer);

      auto *sub_func_seq = dynamic_cast<FunctionSequence *>(function.get());
      if (sub_func_seq != nullptr)
      {
        sub_func_seq->enableDynamicShapeInferer(true);
        sub_func_seq->dynamic_tensor_ctx(dynamic_tensor_ctx());
      }

      // run kernel
      function->run();

      // deallocate input tensors which is no longer used
      _dynamic_tensor_ctx->dynamic_tensor_manager->deallocInput(*op_seq_iter);

      op_seq_iter++;
    }
  }
  else
  {
    for (const auto &function : _functions)
    {
      auto *sub_func_seq = dynamic_cast<FunctionSequence *>(function.get());
      if (sub_func_seq != nullptr)
      {
        sub_func_seq->enableDynamicShapeInferer(false);
      }
      function->run();
    }
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

} // namespace exec
} // namespace onert
