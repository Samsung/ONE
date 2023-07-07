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

#include "ParallelScheduler.h"

#include <cassert>

#include <memory>
#include "util/logging.h"

namespace onert
{
namespace exec
{

ParallelScheduler::ParallelScheduler(const BackendSet &backends)
{
  assert(!backends.empty());

  for (auto &&backend : backends)
  {
    _thread_pools[backend] = std::make_unique<ThreadPool>();
  }
}

void ParallelScheduler::assign(std::unique_ptr<IFunction> &&fn, const backend::Backend *backend)
{
  assert(!_thread_pools.empty());

  _thread_pools.at(backend)->enqueue(std::move(fn));
}

void ParallelScheduler::finish()
{
  for (auto &&itr : _thread_pools)
  {
    itr.second->finish();
  }
}

} // namespace exec
} // namespace onert
