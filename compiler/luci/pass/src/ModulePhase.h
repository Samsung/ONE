/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __MODULE_PHASE_H__
#define __MODULE_PHASE_H__

#include <luci/LuciPass.h>

#include <logo/Phase.h>

#include <vector>

namespace luci
{

using Phase = std::vector<std::unique_ptr<Pass>>;

template <logo::PhaseStrategy S> class PhaseRunner;

template <>
class PhaseRunner<logo::PhaseStrategy::Saturate> final : public logo::PhaseRunnerMixinObservable
{
public:
  PhaseRunner(luci::Module *module) : _module{module}
  {
    // DO NOTHING
  }

public:
  void run(const Phase &) const;

private:
  luci::Module *_module;
};

template <>
class PhaseRunner<logo::PhaseStrategy::Restart> final : public logo::PhaseRunnerMixinObservable
{
public:
  PhaseRunner(luci::Module *module) : _module{module}
  {
    // DO NOTHING
  }

public:
  void run(const Phase &) const;

private:
  luci::Module *_module;
};

} // namespace luci

#endif // __MODULE_PHASE_H__
