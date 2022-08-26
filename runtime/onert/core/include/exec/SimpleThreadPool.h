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

#ifndef __ONERT_EXEC_SIMPLE_THREAD_POOL_H__
#define __ONERT_EXEC_SIMPLE_THREAD_POOL_H__

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace onert
{
namespace exec
{

class SimpleThreadPool
{
public:
  SimpleThreadPool(size_t num_threads);
  ~SimpleThreadPool();

  /**
   * @brief Enqueue a job
   *
   * @tparam F
   * @tparam Args
   * @param f
   * @param args
   * @return std::future<typename std::result_of<F(Args...)>::type>
   */
  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> EnqueueJob(F &&f, Args &&... args)
  {
    if (_stop_all)
    {
      throw std::runtime_error("Stop all threads in SimpleThreadPool");
    }

    using return_type = typename std::result_of<F(Args...)>::type;
    auto job = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    std::future<return_type> job_result_future = job->get_future();
    {
      std::lock_guard<std::mutex> lock(_m_job_q);
      _jobs.push([job]() { (*job)(); });
    }
    _cv_job_q.notify_one();

    return job_result_future;
  }

private:
  /**
   * @brief
   *
   */
  void WorkerThread();

private:
  /**
   * @brief The number of threads
   *
   */
  size_t _num_threads;

  /**
   * @brief Threads worked for jobs
   *
   */
  std::vector<std::thread> _worker_threads;

  /**
   * @brief Queue for jobs
   *
   */
  std::queue<std::function<void()>> _jobs;

  /**
   * @brief condition_variable for the queue _jobs
   *
   */
  std::condition_variable _cv_job_q;

  /**
   * @brief mutex for the queue _jobs
   *
   */
  std::mutex _m_job_q;

  /**
   * @brief Whether all threads are stopped
   *
   */
  bool _stop_all;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_SIMPLE_THREAD_POOL_H__
