/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file  Execution.h
 * @brief This file defines execution
 */
#ifndef __ONERT_EXEC_EXECUTION_H__
#define __ONERT_EXEC_EXECUTION_H__

#include "ExecutionContext.h"
#include "backend/train/ITrainableTensor.h"
#include "exec/IExecutors.h"
#include "ir/Layout.h"

#include <semaphore.h>

#include <deque>
#include <thread>

namespace onert::exec
{

/**
 * @brief Class to define execution instance to collect input/output information for inference
 *        and prepare executor run (TODO)
 */
class Execution
{

public:
  /**
   * @brief     Construct a new Execution object
   * @param[in] executor  Model executor
   */
  Execution(const std::shared_ptr<IExecutors> &executors);

  /**
   * @brief     Construct a new Execution object for signature
   * @param[in] executors   Model executors
   * @param[in] entry_index Entry subgraph index
   */
  Execution(const std::shared_ptr<IExecutors> &executors, const ir::SubgraphIndex &entry_index);

public:
  /**
   * @brief   Returns primary graph object
   * @return  Graph object
   */
  const ir::Graph &primary_subgraph() const { return entryExecutor()->graph(); }

  /**
   * @brief     Change input shape
   * @param[in] index   Input index
   * @param[in] new_shape shape to change
   */
  void changeInputShape(const ir::IOIndex &index, const ir::Shape &new_shape);

  /**
   * @brief     Set input data's information
   * @param[in] index   Input index
   * @param[in] buffer  Input data's buffer pointer
   * @param[in] length  Input data's length
   */
  void setInput(const ir::IOIndex &index, const void *buffer, size_t length);

  /**
   * @brief     Set output data's information
   * @param[in] index   Output index
   * @param[in] buffer  Output data's buffer pointer
   * @param[in] length  Output data's length
   */
  void setOutput(const ir::IOIndex &index, void *buffer, size_t length);

  /**
   * @brief     Get the Input Info object
   * @param[in] index Input index
   * @return    Input info
   */
  const ir::OperandInfo &inputInfo(const ir::IOIndex &index)
  {
    return _ctx.desc.inputs.at(index.value()).info;
  }

  /**
   * @brief     Get the Output Info object
   * @param[in] index Output index
   * @return    Output info
   */
  const ir::OperandInfo &outputInfo(const ir::IOIndex &index)
  {
    return _ctx.desc.outputs.at(index.value()).info;
  }

  /**
   * @brief     Get internally allocated output buffer
   * @param[in] index Output index
   * @return    Buffer pointer
   */
  const void *outputBuffer(const ir::IOIndex &index) const
  {
    return entryExecutor()->outputBuffer(index.value());
  }

  /**
   * @brief   Get input size
   * @return  Input size
   */
  uint32_t inputSize() { return _ctx.desc.inputs.size(); }

  /**
   * @brief   Get output size
   * @return  Output size
   */
  uint32_t outputSize() { return _ctx.desc.outputs.size(); }

  /**
   * @brief  Execution
   * @note   It should be called after setting input and output buffer
   */
  void execute();

  /**
   * @brief Start asynchronous execution
   * @note  It returns after execution thread is started
   *        It should be called after setting input and output buffer
   */
  void startExecute(void);

  /**
   * @brief Return when execution is finished
   * @note  It waits until execution is finished
   */
  void waitFinish(void);

  /**
   * @brief   Check execution is finished
   * @return  @c true if execution is finished, otherwise @c false
   */
  bool isFinished(void) const;

  /**
   * @brief  Train
   * @note   It should be called after setting input and output buffer
   * @param training_step The number of iterations of the training process.
   *                      In other words, the number of gradient update.
   */
  void train(uint32_t training_step);

  /**
   * @brief     Get loss
   * @note      It should be called after training
   * @param[in] ind   Output index
   * @return @c float Loss value
   */
  float getLoss(const ir::IOIndex &ind);

  /**
   * @brief     Iterate trainable tensors
   * @note      It should be called after training
   * @param[in] fn  function to be called with OperandIndex and a pointer to ITrainableTensor
   */
  void iterateTrainableTensors(
    const std::function<void(const ir::OperandIndex &, const backend::train::ITrainableTensor *)>
      &fn) const;

  /**
   * @brief   Get context of execution
   * @return  Execution context
   */
  const ExecutionContext &context() const { return _ctx; }

  /**
   * @brief     Set context of execution at once
   * @param[in] ctx Execution context
   */
  void restoreContext(const ExecutionContext &ctx) { _ctx = ctx; }

  ExecutionOptions &executionOptions() { return _ctx.options; }

private:
  const IExecutor *entryExecutor() const { return _executors->entryExecutor(); };
  IExecutor *entryExecutor() { return _executors->entryExecutor(); };

private:
  const std::shared_ptr<IExecutors> _executors;
  ExecutionContext _ctx;
  std::unique_ptr<std::thread> _exec_thread;
  bool finished{false};
  std::vector<bool> _is_internal_output_tensor;
};

} // namespace onert::exec

#endif // __ONERT_EXEC_EXECUTION_H__
