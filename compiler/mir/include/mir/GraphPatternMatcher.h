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

#ifndef _MIR_GRAPH_PATTERN_MATCHER_H_
#define _MIR_GRAPH_PATTERN_MATCHER_H_

#include "mir/Graph.h"

namespace mir
{

class Operation;

class GraphPatternMatcher
{
public:
  using Predicate = bool(const Operation *);
  explicit GraphPatternMatcher(Graph *g) : _g(g){};

  /**
   * @brief Match an edge with 2 predicates for ends of the edge
   * @param pattern
   * @return Vector of topmost ops of all matches; empty if no mathces are found
   */
  std::vector<std::pair<Operation *, Operation *>> matchEdge(Predicate p1, Predicate p2);

  /**
   * @brief Match a two level tree where the bottommost node has multiple previous nodes
   * @param p1 Predicate for top node
   * @param p2 Predicate for bottom node
   * @return Vector of pairs : all matches; empty if no matches are found
   */
  std::vector<std::pair<std::vector<Operation *>, Operation *>> matchUpBush(Predicate p1,
                                                                            Predicate p2);

private:
  Graph *_g;
};

} // namespace mir

#endif //_MIR_GRAPH_PATTERN_MATCHER_H_
