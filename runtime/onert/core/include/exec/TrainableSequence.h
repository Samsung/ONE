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

#ifndef __ONERT_EXEC_TRAINABLE_SEQUENCE_H__
#define __ONERT_EXEC_TRAINABLE_SEQUENCE_H__

#include <memory>
#include <cassert>
#include <vector>
#include <functional>

#include "exec/FunctionSequence.h"
#include "exec/ITrainableFunction.h"
#include "exec/DynamicShapeInferer.h"
#include "ir/Operations.h"
#include "backend/ITensorRegistry.h"

namespace onert
{
namespace exec
{

class TrainableSequence : public FunctionSequence
{
public:
  template <typename... Args> TrainableSequence(Args &&... args) { initialize(std::move(args)...); }

private:
  void initialize()
  {
    // Template base case : do nothing
  }

  template <typename T, typename... Args> void initialize(std::unique_ptr<T> &&fn, Args &&... args)
  {
    _functions.emplace_back(std::move(fn));
    initialize(std::move(args)...);
  }

public:
  virtual ~TrainableSequence() = default;

  void run() override;
  void prepare() override;

  /**
   * @brief Appends an ITrainableFunction object to the function sequence
   *
   * @param function ITrainableFunction object to be appended
   */
  void append(std::unique_ptr<ITrainableFunction> &&function);

  void iterate(const std::function<void(ITrainableFunction &)> &fn);

  template <typename T, typename... Args> void wrap(Args &&... args)
  {
    for (auto &&function : _functions)
    {
      function = std::make_unique<T>(std::move(function), args...);
    }
  }

protected:
  std::vector<std::unique_ptr<ITrainableFunction>> _functions;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_TRAINABLE_SEQUENCE_H__
