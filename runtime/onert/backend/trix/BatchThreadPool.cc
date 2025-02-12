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

#include "BatchThreadPool.h"

namespace onert::backend::trix
{

BatchThreadPool::BatchThreadPool(size_t num_threads) : _num_threads(num_threads), _stop_all(false)
{
  _worker_threads.reserve(_num_threads);
  for (uint32_t thread_num = 0; thread_num < _num_threads; ++thread_num)
  {
    _worker_threads.emplace_back([this, thread_num]() { this->worker(thread_num); });
  }
}

void BatchThreadPool::worker(uint32_t thread_num)
{
  while (true)
  {
    std::unique_lock<std::mutex> lock(_m_job_queue);
    _cv_job_queue.wait(lock, [this]() { return !this->_job_queue.empty() || _stop_all; });
    if (_stop_all && this->_job_queue.empty())
    {
      return;
    }

    // Pop a job in front of queue
    auto job = std::move(_job_queue.front());
    _job_queue.pop();
    lock.unlock();

    // Run the job
    job(thread_num);
  }
}

BatchThreadPool::~BatchThreadPool()
{
  _stop_all = true;
  _cv_job_queue.notify_all();

  for (auto &&t : _worker_threads)
  {
    t.join();
  }
}

} // namespace onert::backend::trix
