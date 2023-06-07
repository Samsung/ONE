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

#include "exec/TrainableSequence.h"

#include "ir/Operation.h"
#include "backend/ITensorRegistry.h"
#include "util/logging.h"

namespace onert
{
namespace exec
{

void TrainableSequence::run()
{
  for (const auto &function : _functions)
  {
    function->run();
  }
}

void TrainableSequence::prepare()
{
  for (const auto &function : _functions)
  {
    function->prepare();
  }
}

void TrainableSequence::forward(bool training)
{
  for (const auto &function : _trainable_fns)
  {
    function->forward(training);
  }
}

void TrainableSequence::backward()
{
  for (const auto &function : _trainable_fns)
  {
    function->backward();
  }
}

void TrainableSequence::append(std::unique_ptr<ITrainableFunction> &&function)
{
  _trainable_fns.push_back(std::move(function));
}

void TrainableSequence::iterate(const std::function<void(ITrainableFunction &)> &fn)
{
  for (const auto &func : _trainable_fns)
  {
    fn(*func);
  }
}

} // namespace exec
} // namespace onert
