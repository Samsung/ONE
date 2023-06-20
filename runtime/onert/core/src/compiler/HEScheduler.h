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

/**
 * @file  HEScheduler.h
 * @brief This file contains HEScheduler class to define and run task Heterogeneous Execution
 * Scheduler
 */

#ifndef __ONERT_COMPILER_H_E_SCHEDULER_H_
#define __ONERT_COMPILER_H_E_SCHEDULER_H_

#include "IScheduler.h"
#include "../backend/builtin/Config.h"
#include "../exec/ExecTime.h"

#include <backend/Backend.h>
#include <compiler/BackendManager.h>
#include <compiler/Compiler.h>
#include <ir/Graph.h>
#include <ir/OperationIndexMap.h>

#include <map>
#include <memory>

namespace onert
{

namespace compiler
{
/**
 * @brief Class to schedule tasks
 */
class HEScheduler : IScheduler
{
public:
  /**
   * @brief     Construct a new Heterogeneous Execution Scheduler object
   * @param[in] model Graph model
   * @param[in] backend_resolver backend resolver
   */
  HEScheduler(const std::vector<const backend::Backend *> &backends, const CompilerOptions &options)
    : _is_supported{}, _backends_avail_time{}, _ops_eft{},
      _op_to_rank{std::make_shared<ir::OperationIndexMap<int64_t>>()},
      _is_profiling_mode{options.he_profiling_mode}, _is_linear_exec{options.executor == "Linear"},
      _is_parallel_exec{options.executor == "Parallel"}
  {
    for (auto entry : backends)
    {
      if (entry->config()->id() == backend::builtin::Config::ID)
        continue;
      _all_backends.push_back(entry);
    }
    _backend_resolver = std::make_unique<compiler::BackendResolver>();
    _exec_time = std::make_unique<exec::ExecTime>(_all_backends);

    // Find cpu backend
    auto cpu_backend_it =
      std::find_if(_all_backends.begin(), _all_backends.end(), [](const backend::Backend *backend) {
        return backend->config()->id() == "cpu";
      });
    if (cpu_backend_it == _all_backends.end())
      throw std::runtime_error("HEScheduler could be used only if 'cpu' backend is available");
    _cpu_backend = *cpu_backend_it;
  }

public:
  /**
   * @brief   Task scheduling
   *
   * @note    The main idea is taken from HSIP algo:
   *          https://www.hindawi.com/journals/sp/2016/3676149/
   */
  std::unique_ptr<compiler::BackendResolver> schedule(const ir::Graph &graph) final;
  std::shared_ptr<ir::OperationIndexMap<int64_t>> getIndexedRanks() { return _op_to_rank; }

private:
  bool isNodeProfiled(const ir::IOperation &);

  bool schedule(const ir::OperationIndex &, const backend::Backend *parent_backend);
  /**
   * @brief   Get earliest starting time and execution time of an operation on a backend.
   *
   * @note  Returns a time when operation's inputs are ready and backend is available
   *        It also returns exec time. If this is "cpu" backend, then exec_time*CPU_DELAY
   *
   * @param[in] backend: backend, for which to return the time
   * @param[in] index: index of an operation
   * @param[out] transfer_st_exec_time: est and exec time of data transfer operation
   *
   * @return earliest starting time and execution time
   */
  std::pair<int64_t, int64_t>
  ESTAndExecTime(const backend::Backend *backend, const ir::OperationIndex &index,
                 std::multimap<int64_t, int64_t> &transfer_st_exec_time);
  /**
   * @brief   Returns the latest finishing time of parents of a node.
   *
   * @param[in] backend: backend, for which to return the time
   * @param[in] node: node to get eft of parents
   * @param[out] transfer_st_exec_time: est and exec time of data transfer operation
   *
   * @return earliest finishing time of parent nodes
   */
  int64_t predMaxEFT(const backend::Backend *backend, const ir::IOperation &node,
                     std::multimap<int64_t, int64_t> &transfer_st_exec_time);

  void makeRank();

  int64_t DFSMaxRank(const ir::OperationIndex &index);

  int64_t DFSChildrenMaxRank(const ir::OperationIndex &index);
  /**
   * @brief   Returns the time, when backend is available for at least given amount of time.
   *
   * @note  Returns either hole/gap between two performing two already scheduled operations,
   *        or the finishing time of the last scheduled operation
   *
   * @param[in] backend backend, for which to return the time
   * @param[in] starting_time time, starting which to look for gap
   * @param[in] time_amount amount of the time, for which to look gap
   *
   * @return time, when backend has at least time_amount free time
   */
  int64_t backendAvailableTime(const backend::Backend *backend, const int64_t &starting_time,
                               const int64_t &time_amount);

  int64_t getOpTime(const backend::Backend *backend, const std::string &operation, bool quant,
                    uint32_t size);

  int64_t getPermuteTime(const backend::Backend *src_backend, const backend::Backend *dst_backend,
                         bool quant, uint32_t size);

  void scheduleShufflingBackends();

  int64_t tryBackend(const ir::IOperation &node, const backend::Backend *backend);

  /**
   * @brief   Schedule a node and its successor until:
   *            1. there is no branching or connection of multiple branches
   *            2. for subsequent nodes: other than predecessor's backend is prefered
   *
   * @param[in] index: index of an operation
   * @param[in] scheduled: a map to check if this node has already been scheduled
   *
   * @return N/A
   */
  void scheduleBranch(const ir::OperationIndex &index, ir::OperationIndexMap<bool> &scheduled);

private:
  // This variable stores backend/node pairs with unknown execution time, and hints scheduler
  // whether it should assign these backends to these nodes:
  // * It stores false for unsupported nodes
  // * During rank calculation with enabled profiling mode it stores true for supported nodes
  std::unordered_map<const backend::Backend *, std::unordered_map<std::string, bool>> _is_supported;
  // Finishing and starting time of each backend
  std::unordered_map<const backend::Backend *, std::map<int64_t, int64_t>> _backends_avail_time;
  ir::OperationIndexMap<int64_t> _ops_eft;
  std::multimap<int64_t, ir::OperationIndex, std::greater<int64_t>> _rank_to_op;
  std::shared_ptr<ir::OperationIndexMap<int64_t>> _op_to_rank;
  std::unique_ptr<compiler::BackendResolver> _backend_resolver;
  std::unique_ptr<exec::ExecTime> _exec_time;
  const ir::Graph *_graph{nullptr};
  std::vector<const backend::Backend *> _all_backends;
  const backend::Backend *_cpu_backend{nullptr}; // TODO Change this to _builtin_backend
  bool _is_profiling_mode;
  bool _is_linear_exec;
  bool _is_parallel_exec;
};

} // namespace compiler

} // namespace onert

#endif // __ONERT_COMPILER_H_E_SCHEDULER_H_
