/*
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

#include "benchmark/Phases.h"
#include "benchmark/Types.h"
#include "benchmark/MemoryInfo.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <time.h>

namespace
{

uint64_t nowMicros()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_nsec) / 1e3 + static_cast<uint64_t>(ts.tv_sec) * 1e6;
}

void SleepForMicros(uint64_t micros)
{
  timespec sleep_time;
  sleep_time.tv_sec = micros / 1e6;
  micros -= sleep_time.tv_sec * 1e6;
  sleep_time.tv_nsec = micros * 1e3;
  nanosleep(&sleep_time, nullptr);
}
} // namespace

namespace benchmark
{

Phases::Phases(const PhaseOption &option) : _option(option), _mem_before_init(0), _mem_after_run(0)
{
  assert(prepareVmRSS());
  _mem_before_init = getVmHWM();

  if (_option.memory)
  {
    _mem_poll = std::make_unique<MemoryPoller>(std::chrono::milliseconds(option.memory_interval),
                                               option.memory_gpu);
  }
}

void Phases::run(const std::string &tag, const PhaseFunc &exec, const PhaseFunc *post,
                 uint32_t loop_num, bool option_disable)
{
  Phase phase{tag, loop_num};
  PhaseEnum p = getPhaseEnum(tag);
  for (uint32_t i = 0; i < loop_num; ++i)
  {
    if (!option_disable && _option.memory)
      _mem_poll->start(p);

    uint64_t t = 0u;
    t = nowMicros();

    exec(phase, i);

    t = nowMicros() - t;

    if (!option_disable && _option.memory)
      _mem_poll->end(p);

    phase.time.emplace_back(t);

    if (!option_disable && _option.memory)
    {
      phase.memory[MemoryType::RSS].emplace_back(_mem_poll->getRssMap().at(p));
      phase.memory[MemoryType::HWM].emplace_back(_mem_poll->getHwmMap().at(p));
      phase.memory[MemoryType::PSS].emplace_back(_mem_poll->getPssMap().at(p));
    }

    if (post)
      (*post)(phase, i);

    if (_option.run_delay > 0 && p == PhaseEnum::EXECUTE && i != loop_num - 1)
    {
      SleepForMicros(_option.run_delay);
    }
  }

  _mem_after_run = getVmHWM();

  if (p == PhaseEnum::END_OF_PHASE)
  {
    return;
  }

  // TODO Support to store multiple phase data in one tag
  // onert_train can run training function multiple times.
  // So, we need to store multiple phase data in one tag.
  // But, now, it only stores the first phase data.
  if (_phases.find(tag) == _phases.end())
  {
    _phases.emplace(tag, phase);
  }
}

} // namespace benchmark
