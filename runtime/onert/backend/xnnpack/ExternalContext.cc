/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "ExternalContext.h"

#include <cassert>

namespace onert
{
namespace backend
{
namespace xnnpack
{

ExternalContext::ExternalContext(size_t num_threads)
    : _threadpool(pthreadpool_create(num_threads), pthreadpool_destroy), _num_threads(num_threads)
{
  assert(_threadpool);
}

} // namespace xnnpack
} // namespace backend
} // namespace onert
