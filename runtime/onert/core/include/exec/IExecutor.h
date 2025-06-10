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

#include "ExecutionContext.h"
#include "backend/IPortableTensor.h"
#include "ir/Graph.h"
#include "ir/Index.h"
#include "ir/OperandInfo.h"
#include "ir/OperationIndexMap.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace onert::exec
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
   * @brief Execute with given input/output tensors
   *
   * Input and output tensors must be given.
   *
   * @param[in] inputs  Tensors that are passed as inputs
   * @param[in] outputs Tensors that are passed as outputs
   * @param[in] options Execution options
   */
  virtual void execute(const std::vector<backend::IPortableTensor *> &inputs,
                       const std::vector<backend::IPortableTensor *> &outputs,
                       const ExecutionOptions &options) = 0;

  /**
   * @brief   Get input size
   * @return  Input size
   */
  virtual uint32_t inputSize() const = 0;

  /**
   * @brief   Get output size
   * @return  Output size
   */
  virtual uint32_t outputSize() const = 0;

  /**
   * @brief     Get input info at index
   * @param[in] index Index of input
   * @return    Input operand info
   */
  virtual const ir::OperandInfo &inputInfo(uint32_t index) const = 0;

  /**
   * @brief     Get output info at index
   * @param[in] index Index of output
   * @return    Output operand info
   */
  virtual const ir::OperandInfo &outputInfo(uint32_t index) const = 0;

  /**
   * @brief     Get input layout at index
   * @param[in] index Index of input
   * @return    Input operand layout
   */
  virtual ir::Layout inputLayout(uint32_t index) const = 0;

  /**
   * @brief     Get output layout at index
   * @param[in] index Index of output
   * @return    Output operand layout
   */
  virtual ir::Layout outputLayout(uint32_t index) const = 0;

  /**
   * @brief     Get output buffer at index
   * @param[in] index Index of output
   * @return    Output buffer
   */
  virtual const uint8_t *outputBuffer(uint32_t index) const = 0;

  /**
   * @brief   Return current execution configuration
   * @return  Current execution configuration
   */
  virtual const ExecutionOptions &currentOptions() const = 0;
};

} // namespace onert::exec

#endif // __ONERT_EXEC_I_EXECUTOR_H__
