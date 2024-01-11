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

#ifndef __NNFW_BENCHMARK_PHASES_H__
#define __NNFW_BENCHMARK_PHASES_H__

#include "Phase.h"
#include "MemoryPoller.h"

#include <string>
#include <functional>
#include <unordered_map>

namespace benchmark
{

class Phases
{
public:
  Phases(const PhaseOption &option);

  using PhaseFunc = std::function<void(const Phase &, uint32_t)>;

  void run(const std::string &tag, const PhaseFunc &exec, uint32_t loop_num = 1,
           bool option_disable = false)
  {
    run(tag, exec, nullptr, loop_num, option_disable);
  }

  void run(const std::string &tag, const PhaseFunc &exec, const PhaseFunc &post,
           uint32_t loop_num = 1, bool option_disable = false)
  {
    run(tag, exec, &post, loop_num, option_disable);
  }

  const PhaseOption &option() const { return _option; }
  const MemoryPoller &mem_poll() const { return *_mem_poll; }
  const Phase &at(const std::string &tag) const { return _phases.at(tag); }

  uint32_t mem_before_init() const { return _mem_before_init; }
  uint32_t mem_after_run() const { return _mem_after_run; }

private:
  void run(const std::string &tag, const PhaseFunc &exec, const PhaseFunc *post, uint32_t loop_num,
           bool option_disable);

private:
  const PhaseOption _option;
  std::unordered_map<std::string, Phase> _phases;
  std::unique_ptr<MemoryPoller> _mem_poll;
  uint32_t _mem_before_init;
  uint32_t _mem_after_run;
};

} // namespace benchmark

#endif // __NNFW_BENCHMARK_PHASES_H__
