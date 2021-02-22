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

#ifndef __EXAMPLE_GRAPH_H__
#define __EXAMPLE_GRAPH_H__

#include <loco.h>

#include <memory>

namespace
{

enum GraphCode
{
  PullPush, /* Pull - Push network */
};

template <GraphCode Code> struct Bundle;
template <GraphCode Code> std::unique_ptr<Bundle<Code>> make_bundle(void);

template <> struct Bundle<PullPush>
{
  std::unique_ptr<loco::Graph> g;
  loco::Pull *pull;
  loco::Push *push;

  loco::Graph *graph(void) { return g.get(); }
};

template <> std::unique_ptr<Bundle<PullPush>> make_bundle(void)
{
  auto g = loco::make_graph();

  auto pull = g->nodes()->create<loco::Pull>();

  pull->rank(2);
  pull->dim(0) = loco::make_dimension(); // Mark dim 0 as unknown
  pull->dim(1) = 4;

  auto push = g->nodes()->create<loco::Push>();

  push->from(pull);

  auto res = std::make_unique<Bundle<PullPush>>();

  res->g = std::move(g);
  res->pull = pull;
  res->push = push;

  return std::move(res);
}

} // namespace

#endif // __EXAMPLE_GRAPH_H__
