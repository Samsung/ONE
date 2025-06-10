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

#include "WorkQueue.h"

#include <cassert>

namespace onert::exec
{

WorkQueue::~WorkQueue()
{
  {
    std::unique_lock<std::mutex> lock(_mu);
    _state = State::FORCE_FINISHING;
  }
  _cv.notify_all();
}

void WorkQueue::operator()()
{
  while (true)
  {
    std::unique_ptr<IFunction> fn = nullptr;

    {
      std::unique_lock<std::mutex> lock{_mu};
      _cv.wait(lock, [this] {
        return (_state == State::FORCE_FINISHING) || (_state == State::FINISHING) ||
               (_state == State::ONLINE && !_functions.empty());
      });

      if (_state == State::FORCE_FINISHING)
      {
        assert(_functions.empty() && "Terminating with unfinished jobs");
        return;
      }
      else if (_state == State::FINISHING && _functions.empty())
      {
        return;
      }
      else
      {
        assert(((_state == State::FINISHING) || (_state == State::ONLINE)) && !_functions.empty());
        fn = std::move(_functions.front());
        _functions.pop();
      }
    }

    assert(fn);
    fn->run();
  }
}

void WorkQueue::enqueue(std::unique_ptr<IFunction> &&fn)
{
  {
    std::unique_lock<std::mutex> lock{_mu};
    _functions.emplace(std::move(fn));
  }
  _cv.notify_one();
}

void WorkQueue::terminate()
{
  {
    std::unique_lock<std::mutex> lock{_mu};
    _state = State::FORCE_FINISHING;
  }
  _cv.notify_all();
}

void WorkQueue::finish()
{
  {
    std::unique_lock<std::mutex> lock{_mu};
    _state = State::FINISHING;
  }
  _cv.notify_all();
}

uint32_t WorkQueue::numJobsInQueue()
{
  std::unique_lock<std::mutex> lock{_mu};
  return _functions.size();
}

} // namespace onert::exec
