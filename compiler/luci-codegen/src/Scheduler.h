/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef NNCC_SCHEDULER_H
#define NNCC_SCHEDULER_H

#include "SubgraphContext.h"
#include "CodegenOptions.h"

namespace luci_codegen
{

struct SchedulerOptions
{
  SchedulerAlgorithm algorithm;
  int cache_l1_size;
};

class Scheduler
{
public:
  Scheduler(SubgraphContext &subgraph, const SchedulerOptions &options);

  void process();

private:
  SubgraphContext &_subgraph;
  SchedulerOptions _options;
};

} // namespace luci_codegen

#endif //NNCC_SCHEDULER_H
