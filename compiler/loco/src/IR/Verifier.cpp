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

#include "loco/IR/Verifier.h"

#include <set>
#include <cassert>

namespace
{

using namespace loco;

struct GraphVerifier final
{
public:
  GraphVerifier(loco::Graph *graph) : _graph{graph}
  {
    // graph SHOULD NOT BE null
    assert(_graph != nullptr);
  }

public:
  // ErrorListener SHOULD outlive GraphVerifier
  GraphVerifier &enroll(ErrorListener *l)
  {
    if (l != nullptr)
    {
      _listeners.insert(l);
    }
    return (*this);
  }

  GraphVerifier &enroll(std::unique_ptr<ErrorListener> &&l)
  {
    if (l != nullptr)
    {
      _listeners.insert(l.get());
      // Take the ownership of a given listener
      _owned_listeners.insert(std::move(l));
    }
    return (*this);
  }

public:
  void run(void) const
  {
    for (auto node : loco::all_nodes(_graph))
    {
      // Verify nodes
      for (uint32_t n = 0; n < node->arity(); ++n)
      {
        if (node->arg(n) == nullptr)
        {
          notify(ErrorDetail<ErrorCategory::MissingArgument>{node, n});
        }
      }
    }
  }

private:
  template <typename Error> void notify(const Error &error) const
  {
    for (const auto &listener : _listeners)
    {
      listener->notify(error);
    }
  }

private:
  loco::Graph *_graph = nullptr;

  // All active error listeners
  std::set<ErrorListener *> _listeners;

  // Owned error listeners
  std::set<std::unique_ptr<ErrorListener>> _owned_listeners;
};

inline GraphVerifier graph_verifier(loco::Graph *graph) { return GraphVerifier{graph}; }

} // namespace

namespace loco
{

bool valid(Graph *g, std::unique_ptr<ErrorListener> &&l)
{
  class ErrorCounter final : public ErrorListener
  {
  public:
    uint32_t count(void) const { return _count; }

  public:
    void notify(const ErrorDetail<ErrorCategory::MissingArgument> &) { _count += 1; }

  private:
    uint32_t _count = 0;
  };

  ErrorCounter counter;
  graph_verifier(g).enroll(&counter).enroll(std::move(l)).run();
  return counter.count() == 0;
}

} // namespace loco
