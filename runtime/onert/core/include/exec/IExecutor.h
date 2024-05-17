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

/**
 * @file  IExecutor.h
 * @brief This file defines interface of Executor
 */
#ifndef __ONERT_EXEC_I_EXECUTOR_H__
#define __ONERT_EXEC_I_EXECUTOR_H__

#include "ir/Graph.h"
#include "IFunction.h"
#include "ExecutionContext.h"
#include "ir/Index.h"
#include "ir/OperationIndexMap.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace onert
{
namespace backend
{
class IPortableTensor;
namespace builtin
{
class IOTensor;
}
} // namespace backend
} // namespace onert
namespace onert
{
namespace exec
{
/**
 * @brief Struct to define interface of Executor
 */
struct IExecutor
{
  /**
   * @brief Construct a new IExecutor object
   */
  IExecutor() = default;
  /**
   * @brief Destroy the IExecutor object
   */
  virtual ~IExecutor() = default;

  /**
   * @brief Returns graph object
   *
   * @return Graph object
   */
  virtual const ir::Graph &graph() const = 0;

  /**
   * @brief     Set an ordering on operations
   * @param[in] ranks   The table encoding the ordering
   */
  virtual void setIndexedRanks(std::shared_ptr<ir::OperationIndexMap<int64_t>>) = 0;

  /**
   * @brief     Execute with user-given execution context (for primary subgraph)
   * @param[in] ctx Execution context
   * @note      This method should be thread-safe
   */
  virtual void execute(const ExecutionContext &ctx) = 0;

  /**
   * @brief Execute with given input/output tensors
   *
   * For non-primary subgraphs, input and output tensors must be given.
   *
   * @param[in] inputs  tensors that are passed as inputs
   * @param[in] outputs tensors that are passed as outputs
   * @param[in] options Execution options
   */
  virtual void execute(const std::vector<backend::IPortableTensor *> &inputs,
                       const std::vector<backend::IPortableTensor *> &outputs,
                       const ExecutionOptions &options) = 0;

  /**
   * @brief Get input tensor objects
   *
   * @return Vector of @c IOTensor
   */
  virtual const std::vector<backend::builtin::IOTensor *> &getInputTensors() const = 0;

  /**
   * @brief Get output tensor objects
   *
   * @return Vector of @c IOTensor
   */
  virtual const std::vector<backend::builtin::IOTensor *> &getOutputTensors() const = 0;

  /**
   * @brief   Return current execution configuration
   * @return  Current execution configuration
   */
  virtual const ExecutionOptions &currentOptions() const = 0;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_I_EXECUTOR_H__
