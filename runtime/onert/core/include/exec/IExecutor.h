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
#ifndef __ONERT_EXEC_I_EXECUTOR_H_
#define __ONERT_EXEC_I_EXECUTOR_H_

#include "ir/Graph.h"
#include "IFunction.h"
#include "IODescription.h"
#include "ir/OperationIndexMap.h"
#include "backend/IDynamicTensorManager.h"
#include "exec/IFunctionObserver.h"

#define THROW_IF_NOT_OVERRIDEN                 \
  {                                            \
    throw std::runtime_error("Not supported"); \
  }

namespace onert
{
namespace exec
{
class IExecutionObserver;
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
  virtual const ir::Graph &graph() = 0;

  /**
   * @brief     Set an ordering on operations
   * @param[in] ranks   The table encoding the ordering
   */
  virtual void setIndexedRanks(std::shared_ptr<ir::OperationIndexMap<int64_t>>) = 0;

  /**
   * @brief     Start execution
   * @param[in] desc Input and output description
   * @note      This method should be thread-safe
   */
  virtual void execute(const IODescription &desc) = 0;

  // virtual void setOpOutputDumpingInfo(std::unique_ptr<util::OpOutputDumpingInfo>)
  //   THROW_IF_NOT_OVERRIDEN;

  virtual void setFuncObserver(IFunctionObserver *) THROW_IF_NOT_OVERRIDEN;
};

using ExecutorMap = std::unordered_map<ir::SubgraphIndex, std::unique_ptr<IExecutor>>;

// TODO Move this structure to suitable place
/**
 * @brief Dynamic allocation info for input tensors
 *        When user sets shape of input having unknown dims after compilation, memory for the input
 * should be allocated before executing kernels. This struct contains information to allocate
 * memory.
 */
struct DynAllocInfo
{
  /// @brief index of input tensor whose memory needs to be allocated at execution time
  ir::OperandIndex ind;
  /// @brief dynamic tensor manager that can allocate memory when input tensor is dynamic
  backend::IDynamicTensorManager *dyn_tensor_manager;
};

using DynAllocInfoMap = std::unordered_map<std::shared_ptr<backend::ITensor>, DynAllocInfo>;

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_I_EXECUTOR_H_
