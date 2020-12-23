/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __LOGO_PHASE_H__
#define __LOGO_PHASE_H__

#include <logo/Pass.h>

#include <loco.h>

#include <vector>
#include <memory>

namespace logo
{

// Phase is a collection of Pass(es)
using Phase = std::vector<std::unique_ptr<Pass>>;

enum class PhaseEvent
{
  PhaseBegin,
  PhaseEnd,

  PassBegin,
  PassEnd,
};

template <PhaseEvent E> struct PhaseEventInfo;

template <> class PhaseEventInfo<PhaseEvent::PhaseBegin>
{
  // Empty
};

template <> class PhaseEventInfo<PhaseEvent::PhaseEnd>
{
  // Empty
};

template <> class PhaseEventInfo<PhaseEvent::PassBegin>
{
public:
  void pass(const Pass *pass) { _pass = pass; }
  const Pass *pass(void) const { return _pass; }

private:
  const Pass *_pass;
};

template <> class PhaseEventInfo<PhaseEvent::PassEnd>
{
public:
  void pass(const Pass *pass) { _pass = pass; }
  const Pass *pass(void) const { return _pass; }

  void changed(bool changed) { _changed = changed; }
  bool changed(void) const { return _changed; }

private:
  const Pass *_pass;
  bool _changed;
};

struct PhaseEventListener
{
  virtual ~PhaseEventListener() = default;

  virtual void notify(const PhaseEventInfo<PhaseEvent::PhaseBegin> *) { return; };
  virtual void notify(const PhaseEventInfo<PhaseEvent::PhaseEnd> *) { return; };
  virtual void notify(const PhaseEventInfo<PhaseEvent::PassBegin> *) { return; };
  virtual void notify(const PhaseEventInfo<PhaseEvent::PassEnd> *) { return; };
};

// TODO Will be other mix-ins for Phase Runners?
class PhaseRunnerMixinObservable
{
public:
  PhaseRunnerMixinObservable() = default;

public:
  virtual ~PhaseRunnerMixinObservable() = default;

public:
  void attach(PhaseEventListener *listener) { _listener = listener; }

public:
  void notifyPhaseBegin(void) const
  {
    if (_listener)
    {
      PhaseEventInfo<PhaseEvent::PhaseBegin> info;

      _listener->notify(&info);
    }
  }

  void notifyPhaseEnd(void) const
  {
    if (_listener)
    {
      PhaseEventInfo<PhaseEvent::PhaseEnd> info;

      _listener->notify(&info);
    }
  }

  void notifyPassBegin(Pass *pass) const
  {
    if (_listener)
    {
      PhaseEventInfo<PhaseEvent::PassBegin> info;

      info.pass(pass);

      _listener->notify(&info);
    }
  }

  void notifyPassEnd(Pass *pass, bool changed) const
  {
    if (_listener)
    {
      PhaseEventInfo<PhaseEvent::PassEnd> info;

      info.pass(pass);
      info.changed(changed);

      _listener->notify(&info);
    }
  }

private:
  PhaseEventListener *_listener = nullptr;
};

enum class PhaseStrategy
{
  // Run all the passes until there is no pass that makes a change
  Saturate,
  // Same as Saturate but will restart from the first when there is a change
  Restart,
};

template <PhaseStrategy S> class PhaseRunner;

template <> class PhaseRunner<PhaseStrategy::Saturate> final : public PhaseRunnerMixinObservable
{
public:
  PhaseRunner(loco::Graph *graph) : _graph{graph}
  {
    // DO NOTHING
  }

public:
  void run(const Phase &) const;

private:
  loco::Graph *_graph;
};

template <> class PhaseRunner<PhaseStrategy::Restart> final : public PhaseRunnerMixinObservable
{
public:
  PhaseRunner(loco::Graph *graph) : _graph{graph}
  {
    // DO NOTHING
  }

public:
  void run(const Phase &) const;

private:
  loco::Graph *_graph;
};

} // namespace logo

#endif // __LOGO_PHASE_H__
