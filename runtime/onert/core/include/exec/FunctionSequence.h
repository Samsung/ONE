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

public: // context for dynamic tensor
  /**
   * @brief Prepare to run FunctionSequence which "might" handle dynamic tensor
   * @note  Calling this does not mean that run() will handle dynamic tensor.
   *        enableDynamicShapeInferer(true) will make run() will handle dynamic tensor.
   */
  void setDynamicTensorContext(const ir::OpSequence *op_seq, const ir::Operations *operations,
                               std::unique_ptr<exec::DynamicShapeInferer> dynamic_shape_inferer,
                               std::shared_ptr<backend::ITensorRegistry> tensor_registry,
                               backend::IDynamicTensorManager *dynamic_tensor_manager)
  {
    _op_seq = op_seq;
    _operations = operations;
    _dynamic_shape_inferer = std::move(dynamic_shape_inferer);
    _tensor_registry = tensor_registry;
    _dynamic_tensor_manager = dynamic_tensor_manager;
  }

  /**
   * @brief Call this function if this FunctionSequence handles dynamic tensors.
   * @note When dynamic tensors are handled, enableDynamicShapeInferer(true) must be called before
   *       run(). If not called, run() assumes that all tensors are static.
   */
  void enableDynamicShapeInferer(bool enable) { _enable_dynamic_shape_inferer = enable; }

  const ir::OpSequence *opSeq()
  {
    assert(_op_seq);
    return _op_seq;
  }

  const ir::Operations *operations()
  {
    assert(_operations);
    return _operations;
  }

  std::shared_ptr<backend::ITensorRegistry> &tensorRegistry()
  {
    assert(_tensor_registry.get());
    return _tensor_registry;
  }

  backend::IDynamicTensorManager *dynamicTensorManager()
  {
    assert(_dynamic_tensor_manager);
    return _dynamic_tensor_manager;
  }

protected:
  std::vector<std::unique_ptr<IFunction>> _functions;

protected: // context to run this sequence when this sequence may handle dynamic tensors
  bool _enable_dynamic_shape_inferer = false;
  const ir::OpSequence *_op_seq = nullptr;
  const ir::Operations *_operations = nullptr;
  std::unique_ptr<exec::DynamicShapeInferer> _dynamic_shape_inferer = nullptr;
  std::shared_ptr<backend::ITensorRegistry> _tensor_registry = nullptr;
  backend::IDynamicTensorManager *_dynamic_tensor_manager = nullptr;
};

//
// TODO Deprecate this
//
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
