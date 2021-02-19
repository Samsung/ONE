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

#include "ir/Layout.h"
#include "exec/IExecutor.h"
#include "IODescription.h"
#include "../../api/src/nnfw_api_internal.h"

#include <thread>
#include <deque>
#include <semaphore.h>

namespace onert
{
namespace exec
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
  Execution(const std::shared_ptr<ExecutorMap> &executors);

public:
  /**
   * @brief   Returns primary graph object
   * @return  Graph object
   */
  const ir::Graph &primary_subgraph() const { return primary_executor()->graph(); }

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
   * @param[in] layout  Input data's data format
   */
  void setInput(const ir::IOIndex &index, const void *buffer, size_t length,
                ir::Layout layout = ir::Layout::NHWC);

  void CreateNewAsyncDesc();
  void setAsyncInput(const ir::IOIndex &index, const void *buffer, size_t length,
                     ir::Layout layout = ir::Layout::NHWC);
  void setAsyncOutput(const ir::IOIndex &index, void *buffer, size_t length,
                      ir::Layout layout = ir::Layout::NHWC);
  void Async_execute();
  void finish_post();
  void finish_wait();
  void deque_post();
  void deque_wait();
  void input_post();
  void input_wait();
  void set_finish();
  bool is_empty_queue();
  void get_result(std::vector<void *> outputs);

  IODescription* get_async_io_desc();
  void push_async_result(IODescription* io_desc);

  /**
   * @brief     Set input data's information, especially to specify unknown dimensions on model
   * build time.
   * @param[in] index   Input index
   * @param[in] type    Input data's type info
   * @param[in] shape   Input data's shape
   * @param[in] buffer  Input data's buffer pointer
   * @param[in] length  Input data's length
   * @param[in] layout  Input data's data format
   */
  void setInput(const ir::IOIndex &index, const ir::TypeInfo &type, const ir::Shape &shape,
                const void *buffer, size_t length, ir::Layout layout = ir::Layout::NHWC);
  /**
   * @brief     Set output data's information
   * @param[in] index   Output index
   * @param[in] buffer  Output data's buffer pointer
   * @param[in] length  Output data's length
   * @param[in] layout  Output data's data format
   */
  void setOutput(const ir::IOIndex &index, void *buffer, size_t length,
                 ir::Layout layout = ir::Layout::NHWC);
  /**
   * @brief     Set output data's information, especially to specify unknown dimensions on model
   * build time.
   * @param[in] index   Output index
   * @param[in] type    Output data's type info
   * @param[in] shape   Output data's shape
   * @param[in] buffer  Output data's buffer pointer
   * @param[in] length  Output data's length
   * @param[in] layout  Output data's data format
   */
  void setOutput(const ir::IOIndex &index, const ir::TypeInfo &type, const ir::Shape &shape,
                 void *buffer, size_t length, ir::Layout layout = ir::Layout::NHWC);
  /**
   * @brief     Set input data's data format
   * @param[in] index   Input index
   * @param[in] layout  Input data's data format
   */
  void setInputLayout(const ir::IOIndex &index, ir::Layout layout);
  /**
   * @brief     Set output data's data format
   * @param[in] index   Output index
   * @param[in] layout  Output data's data format
   */
  void setOutputLayout(const ir::IOIndex &index, ir::Layout layout);
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

  ir::Shape getInputShape(ir::IOIndex ind) const;
  ir::Shape getOutputShape(ir::IOIndex ind) const;

private:
  const std::unique_ptr<IExecutor> &primary_executor() const
  {
    return _executors->at(ir::SubgraphIndex{0});
  };
  std::unique_ptr<IExecutor> &primary_executor() { return _executors->at(ir::SubgraphIndex{0}); };

private:
  const std::shared_ptr<ExecutorMap> _executors;
  IODescription _io_desc;
  std::deque<IODescription *> _async_io_descs;
  std::deque<IODescription *> _async_result;
  sem_t _async_finish;
  sem_t _async_input_sem;
  sem_t _deque;
  std::unique_ptr<std::thread> _exec_thread;
  bool finished{false};
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTION_H__
