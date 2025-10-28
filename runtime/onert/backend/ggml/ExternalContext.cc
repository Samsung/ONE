/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include <fstream>
#include <string>
#include <thread>
#include <unordered_set>

namespace
{

int32_t countPhysicalCores()
{
#ifdef __linux__
  // Count physical cores, not threads by checking thread_siblings.
  std::unordered_set<std::string> siblings;
  for (uint32_t cpu = 0; cpu < UINT32_MAX; ++cpu)
  {
    std::ifstream thread_siblings("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                                  "/topology/thread_siblings");
    if (!thread_siblings.is_open())
    {
      break;
    }

    std::string line;
    if (std::getline(thread_siblings, line))
    {
      siblings.insert(line);
    }
  }
  if (!siblings.empty())
  {
    return static_cast<int32_t>(siblings.size());
  }
#endif

  unsigned int n_threads = std::thread::hardware_concurrency();
  return n_threads > 0 ? (n_threads <= 4 ? n_threads : n_threads / 2) : 1;
}

} // namespace

namespace onert::backend::ggml
{

ExternalContext::ExternalContext()
  : _max_num_threads(onert::util::getConfigInt(onert::util::config::NUM_THREADS)),
    _ggml_context(std::unique_ptr<ggml_context, decltype(&ggml_free)>(
      ggml_init({.mem_size = 0, .mem_buffer = nullptr, .no_alloc = true}), &ggml_free))
{
  if (_max_num_threads <= -1)
    countPhysicalCores();
}

} // namespace onert::backend::ggml
