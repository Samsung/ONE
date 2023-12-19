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

#ifndef __ONERT_EXEC_FUNCTION_SEQUENCE_H__
#define __ONERT_EXEC_FUNCTION_SEQUENCE_H__

#include <memory>
#include <cassert>
#include <vector>
#include <functional>

#include "exec/IFunction.h"
#include "exec/DynamicShapeInferer.h"
#include "ir/Operations.h"
#include "backend/ITensorRegistry.h"

namespace onert
{
namespace exec
{

class FunctionSequence : public IFunction
{
public:
  template <typename... Args> FunctionSequence(Args &&...args) { initialize(std::move(args)...); }

private:
  void initialize()
  {
    // Template base case : do nothing
  }

  template <typename T, typename... Args> void initialize(std::unique_ptr<T> &&fn, Args &&...args)
  {
    _functions.emplace_back(std::move(fn));
    initialize(std::move(args)...);
  }

public:
  virtual ~FunctionSequence() = default;

  void run() override;
  void prepare() override;

  /**
   * @brief Appends an IFunction object to the function sequence
   *
   * @param function IFunction object to be appended
   */
  void append(std::unique_ptr<IFunction> &&function);

  void iterate(const std::function<void(IFunction &)> &fn);

  template <typename T, typename... Args> void wrap(Args &&...args)
  {
    for (auto &&function : _functions)
    {
      function = std::make_unique<T>(std::move(function), args...);
    }
  }

public: // methods related to dynamic tensor
  struct DynamicTensorCtx
  {
    const ir::IOperation *op = nullptr;
    std::shared_ptr<exec::DynamicShapeInferer> dynamic_shape_inferer = nullptr;
  };

  /**
   * @brief Prepare to run FunctionSequence which "might" handle dynamic tensor
   * @note  Calling this does not mean that run() will handle dynamic tensor.
   *        enableDynamicShapeInferer(true) will make run() will handle dynamic tensor.
   */
  void dynamic_tensor_ctx(std::shared_ptr<DynamicTensorCtx> &dynamic_tensor_ctx)
  {
    _dynamic_tensor_ctx = dynamic_tensor_ctx;
  }

  std::shared_ptr<DynamicTensorCtx> &dynamic_tensor_ctx() { return _dynamic_tensor_ctx; }

  /**
   * @brief Call this function by passing @c true if this FunctionSequence handles dynamic tensors
   *        and should run DynamicShapeInferer. This function can be called multiple times and
   *        if @c false is passed during multiple calls, DynamicShapeInfere will not be run.
   * @note This must be called before run(). If not called, run() assumes that all tensors are
   *       dynamic and DynamicShapeInferer will be run.
   */
  void enableDynamicShapeInferer(bool enable)
  {
    _enable_dynamic_shape_inferer = _enable_dynamic_shape_inferer || enable;
  }

  /**
   * @brief Call this function to initialize vars before running
   * @note When we run a model with static tensor input and then run with dynamic tensor input,
   *       _enable_dynamic_shape_inferer is set to @c false at first run.
   *       Once _enable_dynamic_shape_inferer is set to @c true it cannot be changed to @c false
   *       only with calling enableDynamicShapeInferer(). So initializing it to @c false is
   *       necessary.
   * @todo This is a quick fix. Adding this will increase time for run(). Find way to optimize.
   */
  void initRunning() { _enable_dynamic_shape_inferer = false; }

protected:
  std::vector<std::unique_ptr<IFunction>> _functions;

protected:
  bool _enable_dynamic_shape_inferer = false;

  std::shared_ptr<DynamicTensorCtx> _dynamic_tensor_ctx = nullptr;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_FUNCTION_SEQUENCE_H__
