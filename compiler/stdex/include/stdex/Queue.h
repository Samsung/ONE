/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __STDEX_QUEUE_H__
#define __STDEX_QUEUE_H__

#include <queue>

namespace stdex
{

/**
 * @brief Take the front (= first) element from the queue
 * @note The queue SHOULD have at least one element
 */
template <typename T> T take(std::queue<T> &q)
{
  auto res = q.front();
  q.pop();
  return res;
}

} // namespace stdex

#endif // __STDEX_QUEUE_H__
