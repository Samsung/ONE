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

#ifndef __ONERT_EXEC_OBSREVERS_H__
#define __ONERT_EXEC_OBSREVERS_H__

#include "exec/IFunction.h"
#include "ir/Index.h"
#include "ir/OpSequence.h"
#include "ExecTime.h"
#include "util/ITimer.h"
#include "exec/IExecutor.h"
#include "util/EventCollector.h"
#include "util/EventRecorder.h"
#include "util/EventWriter.h"
#include "util/TracingCtx.h"
#include "util/EventWriter.h"

namespace onert
{
namespace exec
{
class IExecutionObserver
{
public:
  /// @brief Invoked just before model (not individual operation) execution begins
  virtual void handleSubgraphBegin(ir::SubgraphIndex) { return; }

  virtual void handleJobBegin(IExecutor *, ir::SubgraphIndex, const ir::OpSequence *,
                              const backend::Backend *) = 0;
  virtual void handleJobEnd(IExecutor *, ir::SubgraphIndex, const ir::OpSequence *,
                            const backend::Backend *) = 0;

  /// @brief Invoked just after model (not individual operation) execution ends
  virtual void handleSubgraphEnd(ir::SubgraphIndex) { return; }

  virtual ~IExecutionObserver() = default;
};

class ProfileObserver : public IExecutionObserver
{
public:
  explicit ProfileObserver(std::shared_ptr<ExecTime> et, const ir::Graph &graph)
    : _et(std::move(et)), _graph(graph)
  {
  }
  void handleJobBegin(IExecutor *, ir::SubgraphIndex, const ir::OpSequence *,
                      const backend::Backend *) override;
  void handleJobEnd(IExecutor *, ir::SubgraphIndex, const ir::OpSequence *,
                    const backend::Backend *) override;

  void handleSubgraphEnd(ir::SubgraphIndex) override { _et->storeOperationsExecTime(); }

private:
  std::unique_ptr<util::ITimer> _timer;
  std::shared_ptr<ExecTime> _et;
  const ir::Graph &_graph;
};

class TracingObserver : public IExecutionObserver
{
public:
  TracingObserver(const std::string &filepath, const ir::Graph &graph,
                  const util::TracingCtx *tracing_ctx);
  ~TracingObserver();
  void handleSubgraphBegin(ir::SubgraphIndex) override;
  void handleJobBegin(IExecutor *, ir::SubgraphIndex, const ir::OpSequence *,
                      const backend::Backend *) override;
  void handleJobEnd(IExecutor *, ir::SubgraphIndex, const ir::OpSequence *,
                    const backend::Backend *) override;
  void handleSubgraphEnd(ir::SubgraphIndex) override;

private:
  static std::string opSequenceTag(const ir::OpSequence *op_seq, const ir::Operations &operations);

private:
  std::unique_ptr<EventRecorder> _recorder;
  EventCollector _collector;
  const ir::Graph &_graph;
  EventWriter *_event_writer;
  const util::TracingCtx *_tracing_ctx;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_OBSREVERS_H__
