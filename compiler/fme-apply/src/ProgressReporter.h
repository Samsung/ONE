/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __FME_APPLY_PROGRESSREPORTER_H__
#define __FME_APPLY_PROGRESSREPORTER_H__

#include <logo/Phase.h>

#include <loco.h>

#include <luci/IR/Module.h>

namespace fme_apply
{

class ProgressReporter : public logo::PhaseEventListener
{
public:
  ProgressReporter(loco::Graph *graph, logo::PhaseStrategy strategy)
    : _graph{graph}, _strategy{strategy}
  {
    // DO NOTHING
  }

public:
  void notify(const logo::PhaseEventInfo<logo::PhaseEvent::PhaseBegin> *) override;
  void notify(const logo::PhaseEventInfo<logo::PhaseEvent::PhaseEnd> *) override;
  void notify(const logo::PhaseEventInfo<logo::PhaseEvent::PassBegin> *) override;
  void notify(const logo::PhaseEventInfo<logo::PhaseEvent::PassEnd> *) override;

public:
  loco::Graph *graph(void) const { return _graph; }
  logo::PhaseStrategy strategy(void) const { return _strategy; }

private:
  loco::Graph *_graph;
  logo::PhaseStrategy _strategy;
};

} // namespace fme_apply

#endif // __FME_APPLY_PROGRESSREPORTER_H__
