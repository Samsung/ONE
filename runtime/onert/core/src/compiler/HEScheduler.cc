/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "HEScheduler.h"

#include "compiler/BackendResolver.h"
#include "ir/Graph.h"
#include "util/logging.h"

#include <cassert>
#include <cmath>

namespace
{

using namespace onert;

uint32_t getOperationsFlattenedIOSize(const ir::Graph &graph, const ir::IOperation &node)
{
  uint32_t size = 0;
  for (const auto &ind :
       (node.getInputs() | ir::Remove::UNDEFINED) + (node.getOutputs() | ir::Remove::UNDEFINED))
  {
    size += graph.operands().at(ind).info().total_size();
  }
  return size;
}

bool isQuant(const ir::Graph &graph, const ir::IOperation &node)
{
  for (const auto &input : node.getInputs() | ir::Remove::UNDEFINED)
  {
    const auto &obj = graph.operands().at(input);
    if (obj.typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM)
    {
      return true;
    }
  }
  return false;
}

bool isWorkaroundSkip(const ir::Graph &, const backend::Backend *, const ir::IOperation &, bool)
{
  // Now, there is no workaround
  return false;
}

// if a node can be merged into op_seq
bool isMergeable(const ir::Graph &graph, const ir::IOperation &node)
{
  size_t prev_op_cnt = 0;
  for (const auto &input : node.getInputs() | ir::Remove::UNDEFINED)
  {
    // only valid_inputs
    const auto &operand = graph.operands().at(input);
    if (operand.isConstant())
      continue;

    // This operand is output of operation, not weight or bias
    if (operand.getDef().valid())
      ++prev_op_cnt;

    // Current node has multiple inputs as concat or at the beginning of the separated branch
    if (prev_op_cnt > 1 || operand.getUses().size() > 1)
    {
      return false;
    }
  }
  return true;
}

} // namespace

namespace onert
{

namespace compiler
{

void HEScheduler::scheduleShufflingBackends()
{
  VERBOSE(HEScheduler::schedule)
    << "Started task scheduling: uses all backends to get more metrics for data transfer"
    << std::endl;
  size_t backend_ind = 0;
  for (const auto &rank : _rank_to_op)
  {
    VERBOSE(HEScheduler::schedule) << "scheduling (" << rank.second << ")" << std::endl;
    const auto &node = _graph->operations().at(rank.second);
    const bool quant = isQuant(*_graph, node);
    const auto size = getOperationsFlattenedIOSize(*_graph, node);
    for (size_t i = 0;; ++i)
    {
      if (i == _all_backends.size())
      {
        // wasn't able to find backend
        assert(false);
        break;
      }
      if (backend_ind == _all_backends.size())
      {
        backend_ind = 0;
      }
      if (isWorkaroundSkip(*_graph, _all_backends[backend_ind], node, quant))
      {
        ++backend_ind;
        continue;
      }
      const auto exec_time =
        _exec_time->getOperationExecTime(_all_backends[backend_ind], node.name(), quant, size);
      // Scheduling to measure data transfer must be done after measuring all backends separately
      assert(exec_time != _exec_time->NOT_FOUND);
      if (exec_time == _exec_time->getMax())
      {
        ++backend_ind;
        continue;
      }
      _backend_resolver->setBackend(rank.second, _all_backends[backend_ind]);
      VERBOSE(HEScheduler::schedule) << "backend for " << node.name() << " is "
                                     << _all_backends[backend_ind]->config()->id() << std::endl;
      ++backend_ind;
      break;
    }
  }
}

bool HEScheduler::isNodeProfiled(const ir::IOperation &node)
{
  const bool quant = isQuant(*_graph, node);
  const auto size = getOperationsFlattenedIOSize(*_graph, node);
  for (const auto *backend : _all_backends)
  {
    const auto exec_time = _exec_time->getOperationExecTime(backend, node.name(), quant, size);
    if (exec_time == _exec_time->NOT_FOUND)
      return false;
  }
  return true;
}

void HEScheduler::scheduleBranch(const ir::OperationIndex &index,
                                 ir::OperationIndexMap<bool> &scheduled)
{
  auto loc_index = index;
  const backend::Backend *parent_backend = nullptr;
  while (true)
  {
    if (scheduled[loc_index])
    {
      return;
    }
    if (!schedule(loc_index, parent_backend))
    {
      return;
    }
    scheduled[loc_index] = true;
    parent_backend = _backend_resolver->getBackend(loc_index);

    const auto &node = _graph->operations().at(loc_index);
    /* get the only output operand, that is input of the next single operation
     *   and just this nodes output.*/
    if (node.getOutputs().size() != 1)
    {
      return;
    }
    const auto &only_out_operand = _graph->operands().at(*node.getOutputs().begin());
    // One of the last nodes
    if (only_out_operand.getUses().size() == 0)
    {
      return;
    }
    loc_index = *only_out_operand.getUses().begin();
    /* verify, that next node is neither beginning nor ending node of a branch*/
    const auto &next_node = _graph->operations().at(loc_index);
    if (!isMergeable(*_graph, next_node))
    {
      return;
    }
  }
}

std::unique_ptr<compiler::BackendResolver> HEScheduler::schedule(const ir::Graph &graph)
{
  _graph = &graph;
  VERBOSE(HEScheduler::schedule) << "task scheduling started" << std::endl;
  // Make ranks and save in descending order
  makeRank();

  for (const auto *backend : _all_backends)
  {
    _backends_avail_time.emplace(backend, std::map<int64_t, int64_t>{{0, 0}});
  }

  if (_is_profiling_mode)
  {
    // Check if profiling info about all backend/node pairs already exists
    bool all_nodes_are_profiled = true;
    _graph->operations().iterate([&](const ir::OperationIndex &, const ir::IOperation &op) {
      if (all_nodes_are_profiled)
        all_nodes_are_profiled = isNodeProfiled(op);
    });

    // If all nodes are already profiled - schedule backends in such order, so more profiling
    // information about between-backends data transfer could be collected
    if (all_nodes_are_profiled)
    {
      scheduleShufflingBackends();
      VERBOSE(HEScheduler::schedule) << "task scheduling finished" << std::endl;
      return std::move(_backend_resolver);
    }
  }

  ir::OperationIndexMap<bool> visited;
  graph.operations().iterate(
    [&](const ir::OperationIndex &index, const ir::IOperation &) { visited[index] = false; });
  // for each task select the backend with the smallest earliest finishing time(eft)
  for (const auto &rank : _rank_to_op)
  {
    scheduleBranch(rank.second, visited);
  }
  VERBOSE(HEScheduler::schedule) << "task scheduling finished" << std::endl;
  return std::move(_backend_resolver);
}

int64_t HEScheduler::getOpTime(const backend::Backend *backend, const std::string &operation,
                               bool quant, uint32_t size)
{
  const auto time = _exec_time->getOperationExecTime(backend, operation, quant, size);
  if (time != _exec_time->NOT_FOUND)
    return time;

  return _is_supported.at(backend).at(operation) ? 1 : _exec_time->getMax();
}

int64_t HEScheduler::getPermuteTime(const backend::Backend *src_backend,
                                    const backend::Backend *dst_backend, bool quant, uint32_t size)
{
  // TODO Change it to getOperationExecTime()
  const auto time = _exec_time->getPermuteTime(src_backend, dst_backend, quant, size);

  if (time != _exec_time->NOT_FOUND)
    return time;

  // FIXME permute time is not recorded so the control reaches here always
  // Makes the scheduler prefer keeping computations on one backend
  return size / 400;
}

int64_t HEScheduler::tryBackend(const ir::IOperation &node, const backend::Backend *backend)
{
  // if there is no profiling info don't use this backend during scheduling
  if (!_is_profiling_mode)
  {
    VERBOSE(HEScheduler::tryBackend)
      << "Trying to HE schedule while there is no profiling info for " << node.name()
      << " on backend " << backend->config()->id() << ". So this backend won't be used. "
      << std::endl;
    _is_supported[backend][node.name()] = false;
    return _exec_time->getMax();
  }
  auto iter = _is_supported.find(backend);
  if (iter != _is_supported.end())
  {
    auto it2 = iter->second.find(node.name());
    if (it2 != iter->second.end())
    {
      return _is_supported[backend][node.name()] ? 1 : _exec_time->getMax();
    }
  }
  try
  {
    // DO NOTHING

    _is_supported[backend][node.name()] = true;
  }
  catch (std::runtime_error &e)
  {
    _is_supported[backend][node.name()] = false;
  }
  return _is_supported[backend][node.name()] ? 1 : _exec_time->getMax();
}

void HEScheduler::makeRank()
{
  VERBOSE(HEScheduler::makeRank) << "task prioritizing" << std::endl;

  _graph->operations().iterate(
    [&](const ir::OperationIndex &index, const ir::IOperation &) { DFSMaxRank(index); });

  // Check that ranks are calculated for all operations(nodes)
  _graph->operations().iterate([&](const ir::OperationIndex &index, const ir::IOperation &) {
    UNUSED_RELEASE(index);
    assert(_op_to_rank->find(index) != _op_to_rank->end());
  });
  VERBOSE(HEScheduler::makeRank) << "task prioritizing finished" << std::endl;
}

int64_t HEScheduler::DFSMaxRank(const ir::OperationIndex &index)
{
  auto op_to_rank_it = _op_to_rank->find(index);
  if (op_to_rank_it != _op_to_rank->end())
    return op_to_rank_it->second;

  const auto &node = _graph->operations().at(index);
  int64_t rank = 0;
  const bool quant = isQuant(*_graph, node);
  const auto size = getOperationsFlattenedIOSize(*_graph, node);
  auto supported_backends_quantity = static_cast<int64_t>(_all_backends.size());

  const auto max_child_rank = DFSChildrenMaxRank(index);

  // get average exec time of this op
  for (const auto &backend : _all_backends)
  {
    auto exec_time = _exec_time->getOperationExecTime(backend, node.name(), quant, size);
    if (exec_time == _exec_time->NOT_FOUND)
    {
      exec_time = tryBackend(node, backend);
    }
    if (exec_time < _exec_time->getMax())
    {
      rank += exec_time;
    }
    else
    {
      // this operation isn't supported in this backend
      --supported_backends_quantity;
    }
  }
  if (supported_backends_quantity == 0)
  {
    throw std::runtime_error{"Encountered unsupported op: " + node.name()};
  }
  rank /= supported_backends_quantity;

  // get standard deviation
  int64_t std = 0;
  for (const auto backend : _all_backends)
  {
    const auto exec_time = getOpTime(backend, node.name(), quant, size);
    if (exec_time < _exec_time->getMax())
    {
      std += (exec_time - rank) * (exec_time - rank);
    }
  }
  std /= supported_backends_quantity;
  if (std > 0)
  {
    std = static_cast<int>(std::sqrt(std));
    rank *= std;
  }
  rank += max_child_rank;

  assert(rank >= 0);
  _rank_to_op.emplace(rank, index);
  _op_to_rank->emplace(index, rank);
  VERBOSE(HEScheduler::DFSMaxRank)
    << "rank of operation (" << index << ")" << node.name() << " is " << rank << std::endl;

  return rank;
}

int64_t HEScheduler::DFSChildrenMaxRank(const ir::OperationIndex &index)
{
  const auto &node = _graph->operations().at(index);
  int64_t max_child_rank = 0;
  for (const auto &output : node.getOutputs() | ir::Remove::UNDEFINED)
  {
    const auto &operand = _graph->operands().at(output);
    const bool quant = operand.typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM;
    // average data transfer cost of this operand's data
    int64_t avg_transfer_cost = 1;
    for (const auto *backend : _all_backends)
    {
      for (const auto *other_backend : _all_backends)
      {
        if (backend == other_backend)
        {
          continue;
        }
        // TODO Change it to builtin backend
        auto transfer_cost =
          getPermuteTime(backend, other_backend, quant, operand.info().total_size());
        avg_transfer_cost += transfer_cost;
      }
    }
    avg_transfer_cost /= _all_backends.size();
    for (const auto &use : operand.getUses())
    {
      const auto cur_child_rank = DFSMaxRank(use);
      max_child_rank = std::max(max_child_rank, cur_child_rank + avg_transfer_cost);
    }
  }
  return max_child_rank;
}

int64_t HEScheduler::backendAvailableTime(const backend::Backend *backend,
                                          const int64_t &starting_time, const int64_t &time_amount)
{
  const auto &backend_times = _backends_avail_time.at(backend);
  // finishing and starting times of an op, that will come after current op
  auto next_op_fst = backend_times.upper_bound(starting_time);
  // finishing time of an op, that will come before current op
  auto prev_op_ft = starting_time;
  // until reach the "hole/gap", that is enough to run this op
  while (next_op_fst != backend_times.end() && next_op_fst->second - prev_op_ft <= time_amount)
  {
    prev_op_ft = next_op_fst->first + 1;
    ++next_op_fst;
  }
  return prev_op_ft;
}

bool HEScheduler::schedule(const ir::OperationIndex &index, const backend::Backend *parent_backend)
{
  VERBOSE(HEScheduler::schedule) << "scheduling (" << index << ")" << std::endl;
  int64_t eft = std::numeric_limits<int64_t>::max(), selected_exec_time = 0;
  const auto &node = _graph->operations().at(index);

  std::multimap<int64_t, int64_t> selected_transfer_st_exec_time;
  // select the backend with the smallest eft of this task
  const backend::Backend *chosen_backend = nullptr;
  for (const auto *backend : _all_backends)
  {
    std::multimap<int64_t, int64_t> transfer_st_exec_time;
    const auto est_and_et = ESTAndExecTime(backend, index, transfer_st_exec_time);

    if (eft > est_and_et.first + est_and_et.second)
    {
      eft = est_and_et.first + est_and_et.second;
      selected_exec_time = est_and_et.second;
      chosen_backend = backend;
      selected_transfer_st_exec_time = transfer_st_exec_time;
    }
  }

  if (chosen_backend == nullptr)
  {
    throw std::runtime_error{"Fail to choose backend on scheduler"};
  }

  // this is part of a branch and it is assigned another backend
  if (parent_backend && parent_backend != chosen_backend)
  {
    return false;
  }
  for (const auto &it : selected_transfer_st_exec_time)
  {
    auto prev_op_ft = backendAvailableTime(_cpu_backend, it.first, it.second);
    _backends_avail_time[_cpu_backend].insert({prev_op_ft + it.second, prev_op_ft});
  }

  _ops_eft[index] = eft;
  _backends_avail_time[chosen_backend].emplace(eft, eft - selected_exec_time);
  _backend_resolver->setBackend(index, chosen_backend);

  VERBOSE(HEScheduler::schedule) << "backend for " << node.name() << " is "
                                 << chosen_backend->config()->id() << ". Its eft: " << eft
                                 << std::endl;
  return true;
}

std::pair<int64_t, int64_t>
HEScheduler::ESTAndExecTime(const backend::Backend *backend, const ir::OperationIndex &index,
                            std::multimap<int64_t, int64_t> &transfer_st_exec_time)
{
  // Permutation will cause creating a separate op_seq that contains just this permutation node.
  // This isn't needed for Linear executor since it doesn't use op_seqs
  // Number 1 ms is picked experimentally
  int64_t permute_fine = 1000;
  // Multiply cpu operations' exec time by 2 because in parallel executor it might be busy with
  // permutation on other branches or non-nnfw specific tasks and have to wait for it.
  // Number 2 is picked experimentally
  const int64_t CPU_DELAY = 2;
  const auto &node = _graph->operations().at(index);
  const bool quant = isQuant(*_graph, node);
  const auto size = getOperationsFlattenedIOSize(*_graph, node);
  // if this node can be part of a op_seq, then assigning different backend will cause creating
  // another op_seq
  if (isMergeable(*_graph, node))
  {
    permute_fine *= 2;
  }
  if (isWorkaroundSkip(*_graph, backend, node, quant))
  {
    return {_exec_time->getMax(), _exec_time->getMax()};
  }
  // get average exec time of the op on this backend
  auto exec_time = getOpTime(backend, node.name(), quant, size);
  if (backend->config()->id() == "cpu" && _is_parallel_exec)
  {
    exec_time *= CPU_DELAY;
  }

  // get max eft of direct (one level above) predecessors
  auto max_pred_eft = predMaxEFT(backend, node, transfer_st_exec_time);

  int64_t total_transfer_cost = 0;
  std::vector<std::multimap<int64_t, int64_t>::iterator> inserted_permutations;
  // Find free time for data transferring and insert it into backend taskset. This is needed:
  //  1. Time for multiple permutations for this node's input is found correctly
  //  2. If backend==cpu, then free time for this node must come after permutations
  for (auto &&it : transfer_st_exec_time)
  {
    if (_is_parallel_exec)
    {
      it.second *= CPU_DELAY;
    }
    if (!_is_linear_exec)
    {
      it.second += permute_fine;
    }
    total_transfer_cost += it.second;

    const auto prev_op_ft = backendAvailableTime(_cpu_backend, it.first, it.second);

    max_pred_eft = std::max(max_pred_eft, prev_op_ft + it.second);

    const auto tmp = _backends_avail_time[_cpu_backend].emplace(prev_op_ft + it.second, prev_op_ft);
    inserted_permutations.push_back(tmp.first);
  }
  // find the hole/gap, where this op can be put or the finishing time of the last assigned op
  auto prev_op_ft = backendAvailableTime(backend, max_pred_eft, exec_time);

  // Remove inserted permutation from cpu's task set
  for (const auto &it : inserted_permutations)
  {
    _backends_avail_time[_cpu_backend].erase(it);
  }

  /* In case non-parallel executor measure just exec time and data transfer time
   * because EFT(prev_op_ft) is the same for all backends. Since two operations
   * can't be run simultaneously, finish of running operation must be waited for.
   * When an operation starts, all backends are free. So, they need time just for
   * data transfer.*/
  if (!_is_parallel_exec)
  {
    VERBOSE(HEScheduler::ESTAndExecTime)
      << "exec_time of (" << index << ") " << node.name() << " quant==" << quant << " on "
      << backend->config()->id() << " is " << exec_time
      << " microseconds. Data transfer cost: " << total_transfer_cost << std::endl;

    return {total_transfer_cost, exec_time};
  }
  VERBOSE(HEScheduler::ESTAndExecTime)
    << "exec_time of (" << index << ") " << node.name() << " quant==" << quant << " on "
    << backend->config()->id() << ": " << exec_time
    << " microseconds. Backend available time: " << prev_op_ft
    << " Parent's max eft: " << max_pred_eft - total_transfer_cost
    << " data transfer cost: " << total_transfer_cost << std::endl;

  return {prev_op_ft, exec_time};
}

int64_t HEScheduler::predMaxEFT(const backend::Backend *backend, const ir::IOperation &node,
                                std::multimap<int64_t, int64_t> &transfer_st_exec_time)
{
  int64_t max_pred_eft = 0;
  for (const auto &input_operand_idx : node.getInputs() | ir::Remove::UNDEFINED)
  {
    const auto &input_operand = _graph->operands().at(input_operand_idx);
    const bool quant = input_operand.typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM;

    auto input_node_idx = input_operand.getDef();
    if (input_node_idx.valid())
    {
      // Data transfer cost from parent's node backend to current node's backend:
      auto parent_backend = _backend_resolver->getBackend(input_node_idx);

      max_pred_eft = std::max(max_pred_eft, _ops_eft.at(input_node_idx));
      if (parent_backend != backend)
      {
        // Multiply operand size by 2 because size must describe input+output size
        int64_t transfer_cost =
          getPermuteTime(parent_backend, backend, quant, input_operand.info().total_size() * 2);
        transfer_st_exec_time.emplace(_ops_eft.at(input_node_idx), transfer_cost);
      }
    }
  }
  return max_pred_eft;
}

} // namespace compiler

} // namespace onert
