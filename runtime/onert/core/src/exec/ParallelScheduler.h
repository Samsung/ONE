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

#ifndef __ONERT_EXEC_PARALLEL_SCHEDULER_H__
#define __ONERT_EXEC_PARALLEL_SCHEDULER_H__

#include <unordered_map>
#include <memory>

#include "exec/IFunction.h"
#include "BackendSet.h"
#include "ThreadPool.h"

namespace onert
{
namespace exec
{

class ParallelScheduler
{
public:
  /**
   * @brief Constructs ParallelScheduler object
   *
   * @param backends Backend set
   */
  ParallelScheduler(const BackendSet &backends);
  /**
   * @brief Assign a task to the given backend
   *
   * @param[in] fn Function to be assigned
   * @param[in] fn Target backend
   */
  void assign(std::unique_ptr<IFunction> &&fn, const backend::Backend *backend);
  /**
   * @brief Block until all jobs are finished
   */
  void finish();

private:
  std::unordered_map<const backend::Backend *, std::unique_ptr<ThreadPool>> _thread_pools;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_PARALLEL_SCHEDULER_H__
