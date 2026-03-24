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

#ifndef NNCC_OPTIMIZATION_UTILS_H
#define NNCC_OPTIMIZATION_UTILS_H

#include "mir/Operation.h"
#include "mir/Graph.h"

namespace nnc
{
namespace opt_util
{
/**
 * @brief Swap adjacent nodes in Graph. Creates new nodes and replaces the old ones with new.
 * @param g MIR Graph
 * @param top Node
 * @param bottom Node
 */
void swapAdjacent(mir::Graph *g, mir::Operation *top, mir::Operation *bottom);

// TODO: this function and it's usages should be removed, after DCE optimization will be implemented
void removeNodeIfUnused(mir::Graph *g, mir::Operation *op);
} // namespace opt_util
} // namespace nnc

#endif // NNCC_OPTIMIZATION_UTILS_H
