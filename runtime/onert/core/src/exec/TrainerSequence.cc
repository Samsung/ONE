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

#include "exec/TrainerSequence.h"

#include "ir/Operation.h"
#include "backend/ITensorRegistry.h"
#include "util/logging.h"
#include "util/ConfigSource.h"

namespace onert
{
namespace exec
{

void TrainerSequence::run()
{
  auto training = util::getConfigBool(util::config::TRAINING_MODE);
  if (training)
  {
    // TODO Consider dynamic tensor
    for (const auto &function : _functions)
    {
      function->forward(training);
    }
    // TODO Calculate loss
    for (auto it = _functions.rbegin(); it != _functions.rend(); ++it)
    {
      it->get()->backward();
    }
  }
  else
  {
    for (const auto &function : _functions)
    {
      function->forward(training);
    }
  }
}

void TrainerSequence::prepare()
{
  // NOTE Do we need `prepare()`?
  // for (const auto &function : _functions)
  // {
  //   function->prepare();
  // }
}

void TrainerSequence::append(std::unique_ptr<ITrainableFunction> &&function)
{
  _functions.push_back(std::move(function));
}

void TrainerSequence::iterate(const std::function<void(ITrainableFunction &)> &fn)
{
  for (const auto &func : _functions)
  {
    fn(*func);
  }
}

} // namespace exec
} // namespace onert
