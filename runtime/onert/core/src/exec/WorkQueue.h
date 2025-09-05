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

#ifndef __ONERT_EXEC_WORK_QUEUE_H__
#define __ONERT_EXEC_WORK_QUEUE_H__

#include "exec/IFunction.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

namespace onert::exec
{

class WorkQueue
{
public:
  enum class State
  {
    ONLINE,
    FINISHING,
    FORCE_FINISHING
  };

public:
  /**
   * @brief Create WorkQueue object
   */
  WorkQueue() = default;
  /**
   * @brief Destroy WorkQueue object
   */
  ~WorkQueue();
  /**
   * @brief Thread entry function
   */
  void operator()();
  /**
   * @brief Push the given Task to the job queue
   *
   * @param fn Function to be executed(a job)
   */
  void enqueue(std::unique_ptr<IFunction> &&fn);
  /**
   * @brief Flag as terminating so all the worker threads can terminate
   */
  void terminate();
  /**
   * @brief Flag as terminating so all the worker threads can terminate
   */
  void finish();
  /**
   * @brief Check if it has pending jobs. Even if this returns fals, WorkQueue threads may be still
   * running
   *
   * @return true if the job queue not empty otherwise false
   */
  uint32_t numJobsInQueue();

private:
  State _state{State::ONLINE};
  std::queue<std::unique_ptr<IFunction>> _functions;
  std::mutex _mu;
  std::condition_variable _cv;
};

} // namespace onert::exec

#endif // __ONERT_EXEC_WORK_QUEUE_H__
