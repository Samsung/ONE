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

#ifndef __ONERT_EXEC_THREAD_POOL_H__
#define __ONERT_EXEC_THREAD_POOL_H__

#include <thread>
#include <memory>
#include <vector>

#include "WorkQueue.h"

namespace onert::exec
{

class ThreadPool
{
public:
  /**
   * @brief Coustruct ThreadPool object
   *
   * @param num_threads Number of threads
   */
  ThreadPool(uint32_t num_threads = 1);
  /**
   * @brief Destroy ThreadPool object
   */
  ~ThreadPool();
  /**
   * @brief Enqueue a function
   *
   * @param fn A function to be queued
   */
  void enqueue(std::unique_ptr<IFunction> &&fn);
  /**
   * @brief Get number of jobs in worker's queue
   *
   * @return Number of jobs
   */
  uint32_t numJobsInQueue();

  /**
   * @brief Block until all jobs are finished
   */
  void finish();

private:
  void join();

private:
  WorkQueue _worker;
  std::vector<std::thread> _threads;
};

} // namespace onert::exec

#endif // __ONERT_EXEC_THREAD_POOL_H__
