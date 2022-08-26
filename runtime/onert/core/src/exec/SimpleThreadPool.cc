/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "exec/SimpleThreadPool.h"

namespace onert
{
namespace exec
{

SimpleThreadPool::SimpleThreadPool(size_t num_threads) : _num_threads(num_threads), _stop_all(false)
{
  _worker_threads.reserve(_num_threads);
  for (size_t i = 0; i < _num_threads; ++i)
  {
    _worker_threads.emplace_back([this]() { this->WorkerThread(); });
  }
}

SimpleThreadPool::~SimpleThreadPool()
{
  _stop_all = true;
  _cv_job_q.notify_all();

  for (auto &t : _worker_threads)
  {
    t.join();
  }
}

void SimpleThreadPool::WorkerThread()
{
  while (true)
  {
    std::unique_lock<std::mutex> lock(_m_job_q);
    _cv_job_q.wait(lock, [this]() { return !this->_jobs.empty() || _stop_all; });
    if (_stop_all && this->_jobs.empty())
    {
      return;
    }

    // Pop a job in front of queue
    std::function<void()> job = std::move(_jobs.front());
    _jobs.pop();
    lock.unlock();

    // Run the job
    job();
  }
}

} // namespace exec
} // namespace onert
