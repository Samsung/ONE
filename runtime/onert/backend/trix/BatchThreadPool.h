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

#ifndef __ONERT_BACKEND_TRIX_BATCH_THREAD_POOL_H__
#define __ONERT_BACKEND_TRIX_BATCH_THREAD_POOL_H__

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace onert
{
namespace backend
{
namespace trix
{

/**
 * @brief Class that has a threadpool for batch-by-batch multi-threading
 *
 */
class BatchThreadPool
{
public:
  BatchThreadPool(size_t num_threads);
  ~BatchThreadPool();

  /**
   * @brief
   *
   * @tparam F    Type of the function for job
   * @tparam Args Type of arguments  of job
   * @param f     Function for job
   * @param args  Arguments of job
   * @return std::future<typename std::result_of<F(uint32_t, Args...)>::type>
   */
  template <class F, class... Args>
  std::future<typename std::result_of<F(uint32_t, Args...)>::type> enqueueJob(F &&f, Args &&...args)
  {
    if (_stop_all)
    {
      throw std::runtime_error("Stop all threads in BatchThreadPool");
    }

    using return_type = typename std::result_of<F(uint32_t, Args...)>::type;
    auto job = std::make_shared<std::packaged_task<return_type(uint32_t)>>(
      std::bind(std::forward<F>(f), std::placeholders::_1, std::forward<Args>(args)...));
    std::future<return_type> job_result_future = job->get_future();
    {
      // Push job in the assigned queue
      std::lock_guard<std::mutex> lock(_m_job_queue);

      // Push job
      _job_queue.push([job](uint32_t thread_num) { (*job)(thread_num); });
    }
    _cv_job_queue.notify_one();

    return job_result_future;
  }

private:
  /**
   * @brief Worker to run jobs
   *
   * @param thread_num Thread number on which worker is running
   */
  void worker(uint32_t thread_num);

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
  std::queue<std::function<void(uint32_t)>> _job_queue;

  /**
   * @brief condition_variables for _job_queue and _worker_threads
   *
   */
  std::condition_variable _cv_job_queue;

  /**
   * @brief Mutex for the queue _job_queue
   *
   */
  std::mutex _m_job_queue;

  /**
   * @brief Whether all threads are stopped
   *
   */
  bool _stop_all;
};

} // namespace trix
} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_TRIX_BATCH_THREAD_POOL_H__
