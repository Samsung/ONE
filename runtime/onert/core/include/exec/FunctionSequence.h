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
#include <vector>
#include <functional>

#include "exec/IFunction.h"
#include "exec/DynamicShapeInference.h"
#include "ir/Operations.h"

#include <memory>

namespace onert
{
namespace exec
{

class FunctionSequence : public IFunction
{
public:
  template <typename... Args> FunctionSequence(Args &&... args) { initialize(std::move(args)...); }

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

  template <typename T, typename... Args> void wrap(Args &&... args)
  {
    for (auto &function : _functions)
    {
      function = std::make_unique<T>(std::move(function), args...);
    }
  }

protected:
  std::vector<std::unique_ptr<IFunction>> _functions;
};

/**
 * @brief Function sequence used for backend that supports dynamic tensor
 *        Such backend cannot use class FunctionSequence but use this class
 */
class FunctionSequenceForDynamicBackend : public FunctionSequence
{
public:
  FunctionSequenceForDynamicBackend(const ir::OpSequence &op_seq, const ir::Operations &operations,
                                    std::unique_ptr<DynamicShapeInferer> dyn_shape_inferer,
                                    backend::IDynamicTensorManager *dyn_tensor_manager)
      : _op_seq(op_seq), _operations_ctx(operations),
        _dyn_shape_inferer(std::move(dyn_shape_inferer)), _dyn_tensor_manager(dyn_tensor_manager)
  { /* empty */
  }

  void run() override;

private:
  const ir::OpSequence &_op_seq;
  const ir::Operations &_operations_ctx;
  /// @brief shape inferer at execution time
  std::unique_ptr<DynamicShapeInferer> _dyn_shape_inferer;
  backend::IDynamicTensorManager *_dyn_tensor_manager;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_FUNCTION_SEQUENCE_H__
