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
#include "backend/ITensorRegistry.h"
#include "util/logging.h"

namespace onert
{
namespace exec
{

void FunctionSequence::run()
{
  if (_enable_dynamic_shape_inferer && _dynamic_tensor_ctx)
  {
    // acl_cl and acl_neon backend don't support dynamic shape.
    // _dynamic_tensor_ctx is always nullptr for acl_cl and acl_neon
    // Thus, those two bakends cannot reach here.

    // Do dynamic shape inference
    _dynamic_tensor_ctx->op->accept(*_dynamic_tensor_ctx->dynamic_shape_inferer);

    for (const auto &function : _functions)
    {
      // NOTE the function could be also FunctionSequence so we do this
      // TODO Remove this or do this recursively
      auto *sub_func_seq = dynamic_cast<FunctionSequence *>(function.get());
      if (sub_func_seq != nullptr)
      {
        sub_func_seq->enableDynamicShapeInferer(true);
        sub_func_seq->dynamic_tensor_ctx(dynamic_tensor_ctx());
      }

      // run kernel
      function->run();
    }
  }
  else
  {
    for (const auto &function : _functions)
    {
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

void FunctionSequence::forward(bool training)
{
  for (const auto &function : _trainable_fns)
  {
    function->forward(training);
  }
}

void FunctionSequence::backward()
{
  for (const auto &function : _trainable_fns)
  {
    function->backward();
  }
}

void FunctionSequence::append(std::unique_ptr<ITrainableFunction> &&function)
{
  _trainable_fns.push_back(std::move(function));
}

void FunctionSequence::iterate(const std::function<void(ITrainableFunction &)> &fn)
{
  for (const auto &func : _trainable_fns)
  {
    fn(*func);
  }
}

} // namespace exec
} // namespace onert
