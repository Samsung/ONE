/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "exec/train/TrainableFnSequence.h"

namespace onert
{
namespace exec
{
namespace train
{

void TrainableFnSequence::forward(bool training)
{
  for (const auto &function : _functions)
  {
    function->forward(training);
  }
}

void TrainableFnSequence::backward(uint32_t training_step)
{
  for (auto it = _functions.rbegin(); it != _functions.rend(); ++it)
  {
    (*it)->backward(training_step);
  }
}

void TrainableFnSequence::append(std::unique_ptr<ITrainableFunction> &&function)
{
  _functions.push_back(std::move(function));
}

void TrainableFnSequence::iterate(const std::function<void(ITrainableFunction &)> &fn)
{
  for (const auto &func : _functions)
  {
    fn(*func);
  }
}

} // namespace train
} // namespace exec
} // namespace onert
