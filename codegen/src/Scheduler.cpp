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

#include "Scheduler.h"
#include "Utilities.h"

namespace luci_codegen
{

Scheduler::Scheduler(luci_codegen::SubgraphContext &subgraph,
                     const luci_codegen::SchedulerOptions &options) : _subgraph(subgraph), _options(options)
{
  static bool initialized = false;
  if (!initialized)
  {
    Halide::load_plugin("autoschedule_adams2019");
    Halide::load_plugin("autoschedule_li2018");
    Halide::load_plugin("autoschedule_mullapudi2016");
  }
  initialized = true;
}

void Scheduler::process()
{
  Halide::Pipeline pipeline = _subgraph.get_pipeline();

  std::vector<Halide::Argument> inputs;
  for (auto subgraph_input: _subgraph.get_inputs())
  {
    inputs.push_back(subgraph_input.second);
  }

  if (_options.algorithm == SchedulerAlgorithm::None)
  {
    return;
  }

  // proceed if need to schedule

  for (auto &input: _subgraph.get_inputs())
  {
    luci::CircleNode *node = input.first;
    Halide::ImageParam input_param = input.second;
    Halide::Region estimates;
    uint32_t rank = node->rank();
    for (int i = 0; i < rank; ++i)
    {
      estimates.emplace_back(0, static_cast<int>(node->dim(rank - 1 - i).value()));
    }
    input_param.set_estimates(std::move(estimates));
  }
  std::vector<Halide::Func> halide_outputs;
  for (auto &output : _subgraph.get_outputs())
  {
    luci::CircleNode *node = output.first;
    Halide::Func output_func = output.second;
    halide_outputs.push_back(output_func);
    Halide::Region estimates;
    uint32_t rank = node->rank();
    for (int i = 0; i < rank; ++i)
    {
      estimates.emplace_back(0, static_cast<int>(node->dim(rank - 1 - i).value()));
    }
    output_func.set_estimates(std::move(estimates));
  }

  Halide::MachineParams params(1, _options.cache_l1_size, 40);

  std::string scheduler_algorithm;
  switch (_options.algorithm)
  {
    case SchedulerAlgorithm::Mullapudi:
      scheduler_algorithm = "Mullapudi2016";
      break;
    case SchedulerAlgorithm::Li:
      scheduler_algorithm = "Li2018";
      break;
    case SchedulerAlgorithm::Adams:
      scheduler_algorithm = "Adams2019";
      break;
    default:
      assert(false && "unsupported scheduling algorithm");
      break;
  }
  Halide::AutoSchedulerResults schedule = pipeline.auto_schedule(scheduler_algorithm, _subgraph.get_target(), params);

  _subgraph.set_schedule(schedule);
}

} // namespace luci_codegen
