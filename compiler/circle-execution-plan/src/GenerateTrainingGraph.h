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

#ifndef GENERATE_TRAINING_GRAPH
#define GENERATE_TRAINING_GRAPH

#include <luci/IR/Module.h>
#include <luci/Plan/CircleNodeExecutionPlan.h>

namespace training_graph
{

class GenerateTrainingGraph
{
public:
  GenerateTrainingGraph(luci::Module *module): _module(module) {}

  std::unique_ptr<loco::Graph> createTrainingGraph(const std::vector<uint32_t> &nodes_ind);

  std::map<uint32_t, uint32_t> createMapTensorsIndexes(const circle::Model *origin, const circle::Model *train);

private:
  luci::Module *_module;
};

} // namespace training_graph

#endif // GENERATE_TRAINING_GRAPH
