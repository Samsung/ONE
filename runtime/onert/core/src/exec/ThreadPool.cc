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

#include "ThreadPool.h"

#include <cassert>

namespace onert::exec
{

ThreadPool::ThreadPool(uint32_t num_threads)
{
  assert(num_threads >= 1);

  for (uint32_t i = 0; i < num_threads; i++)
  {
    _threads.emplace_back(std::ref(_worker));
  }
}

ThreadPool::~ThreadPool()
{
  if (!_threads.empty())
  {
    _worker.terminate();
    join();
  }
}

void ThreadPool::enqueue(std::unique_ptr<IFunction> &&fn) { _worker.enqueue(std::move(fn)); }

uint32_t ThreadPool::numJobsInQueue() { return _worker.numJobsInQueue(); }

void ThreadPool::join()
{
  for (auto &&thread : _threads)
  {
    thread.join();
  }
  _threads.clear();
}

void ThreadPool::finish()
{
  _worker.finish();
  join();
}

} // namespace onert::exec
