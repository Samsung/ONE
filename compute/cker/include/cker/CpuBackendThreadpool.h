/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

#ifndef __NNFW_CKER_CPU_BACKEND_THREADPOOL_H_
#define __NNFW_CKER_CPU_BACKEND_THREADPOOL_H_

#include <ruy/context.h>     // from @ruy
#include <ruy/thread_pool.h> // from @ruy

namespace nnfw
{
namespace cker
{
namespace cpu_backend_threadpool
{

using Task = ruy::Task;

template <typename TaskType>
void Execute(int tasks_count, TaskType *tasks, ruy::Context *ruy_context)
{
  assert(tasks_count <= ruy_context->max_num_threads());
  ruy_context->mutable_thread_pool()->Execute(tasks_count, tasks);
}

} // namespace cpu_backend_threadpool
} // namespace cker
} // namespace nnfw

#endif // __NNFW_CKER_CPU_BACKEND_THREADPOOL_H_
