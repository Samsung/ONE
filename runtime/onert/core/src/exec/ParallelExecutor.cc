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

#include "ParallelExecutor.h"

#include <cassert>

#include "util/logging.h"
#include "exec/IFunction.h"

namespace onert
{
namespace exec
{

class HookFunction : public IFunction
{
public:
  HookFunction(IFunction *fn, const std::function<void()> &setup,
               const std::function<void()> &teardown)
      : _fn{fn}, _setup{setup}, _teardown{teardown}
  {
  }

public:
  void run() override
  {
    _setup();
    _fn->run();
    _teardown();
  }

private:
  IFunction *_fn;
  std::function<void()> _setup;
  std::function<void()> _teardown;
};

void ParallelExecutor::notify(uint32_t finished_job_id)
{
  std::unique_lock<std::mutex> lock{_mu_jobs};

  DataflowExecutor::notify(finished_job_id);

  lock.unlock();
  _cv_jobs.notify_all();
}

ParallelExecutor::ParallelExecutor(std::unique_ptr<compiler::LoweredGraph> lowered_graph,
                                   const compiler::TensorRegistries &tensor_regs,
                                   compiler::CodeMap &&code_map)
    : DataflowExecutor{std::move(lowered_graph), tensor_regs, std::move(code_map)}
{
  VERBOSE(ParallelExecutor) << "Constructing Parallel Executor" << std::endl;
}

void ParallelExecutor::executeImpl()
{
  bool dynamic_input_exists = hasDynamicInput();

  // Init scheduler
  // TODO Consider to have distinct backend set in LowerInfoMap
  BackendSet backends;
  for (auto &itr : _lowered_graph->getLowerInfo()->op_seq)
  {
    backends.add(itr.second->backend());
  }
  _scheduler = std::make_unique<ParallelScheduler>(backends);

  assert(noWaitingJobs());

  // Execution setup
  _waiting_jobs.swap(_finished_jobs); // Move finished jobs to waiting jobs

  for (uint32_t i = 0; i < _waiting_jobs.size(); ++i)
  {
    VERBOSE(ParallelExecutor) << i << ": " << _input_info[i] << std::endl;
    if (_input_info[i] == 0)
    {
      emplaceToReadyJobs(i);
    }
  }
  assert(!_ready_jobs.empty()); // Cannot begin if there is no initial jobs

  VERBOSE(ParallelExecutor) << "INITIAL JOBS : " << _ready_jobs.size() << std::endl;

  _subject.notifyModelBegin(this);
  while (true)
  {
    std::unique_lock<std::mutex> lock{_mu_jobs};

    if (_ready_jobs.empty())
    {
      _cv_jobs.wait(lock, [this] { return !_ready_jobs.empty() || noWaitingJobs(); });
      // Check finish condition
      if (_ready_jobs.empty() && noWaitingJobs())
      {
        break;
      }
    }

    auto job = std::move(_ready_jobs.begin()->second);
    _ready_jobs.erase(_ready_jobs.begin());

    lock.unlock();

    VERBOSE(ParallelExecutor) << "Assigning fn #" << job->index() << std::endl;

    auto job_index = job->index();
    auto op_sequence_index = _job_to_op_seq[job_index];
    auto op_seq = &_lowered_graph->op_seqs().at(op_sequence_index);
    auto backend = _lowered_graph->getLowerInfo()->op_seq.at(op_sequence_index)->backend();
    auto setup = [&, op_seq, backend]() { _subject.notifyJobBegin(this, op_seq, backend); };
    auto teardown = [&, job_index, op_seq, backend]() {
      _subject.notifyJobEnd(this, op_seq, backend);
      notify(job_index);
    };

    job->fn_seq()->initRunning();

    // dynamic tensor setting
    bool handle_dynamic_tensor = op_seq->has_dynamic_tensor() || dynamic_input_exists;
    job->fn_seq()->enableDynamicShapeInferer(handle_dynamic_tensor);

    _scheduler->assign(std::make_unique<HookFunction>(job->fn_seq(), setup, teardown), backend);
    _finished_jobs[job_index] = std::move(job);
  }

  assert(noWaitingJobs());

  // Wait for all the jobs done
  _scheduler->finish();
  _subject.notifyModelEnd(this);

  // Reset input info for the next execution
  _input_info = _initial_input_info;
}

} // namespace exec
} // namespace onert
