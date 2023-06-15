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

#include "ExecutionObservers.h"

#include "../util/EventWriter.h"

#include "util/logging.h"

#include <misc/polymorphic_downcast.h>

#include <string>
#include <sstream>

namespace
{

void setUserData(const onert::ir::Graph &g, const onert::ir::IOperation *op,
                 decltype(EventCollector::Event::userData) &data)
{
  // From a tensor of shape [a, b, c], this will return a string "shape(a b c)".
  // String like "[1, 2, 3]" looks better but this will be considered as a list in Json
  // so text search (e.g., Ctrl-F in Chrome Tracing) could be difficult
  auto build_shape_str = [&](onert::ir::OperandIndex operand_idx) {
    std::string shape_str;
    auto &shape = g.operands().at(operand_idx).info().shape();
    for (int i = 0; i < shape.rank(); i++)
    {
      if (i == 0)
        shape_str = "shape(" + std::to_string(shape.dim(i));
      else
        shape_str += " " + std::to_string(shape.dim(i));
    }
    shape_str += ")";

    return shape_str;
  };

  auto &inputs = op->getInputs();
  auto size = inputs.size();
  for (size_t i = 0; i < size; i++)
  {
    auto operand_idx = inputs.at(i);
    if (operand_idx.undefined())
      continue;

    std::string key("input_shape_" + std::to_string(i));
    std::string value = build_shape_str(operand_idx);
    data.emplace_back(std::make_pair(key, value));
  }

  // add other userData as needed
}

} // namespace

namespace onert
{

namespace exec
{

void ProfileObserver::handleJobBegin(onert::exec::IExecutor *, ir::SubgraphIndex,
                                     ir::OperationIndex, const onert::backend::Backend *backend)
{
  _timer = backend->config()->timer();
  if (_timer == nullptr)
    throw std::runtime_error("To profile backend timer() method must be implemented");
  _timer->handleBegin();
}

void ProfileObserver::handleJobEnd(IExecutor *exec, ir::SubgraphIndex,
                                   const ir::OperationIndex op_ind, const backend::Backend *backend)
{
  _timer->handleEnd();
  const auto timer_res = _timer->getTime();

  // NOTE This assumes there is just one operation in a op
  const auto &node = _graph.operations().at(op_ind);
  auto node_name = node.name();
  VERBOSE(ProfileInfo) << "Time for " << node_name << " : " << timer_res << std::endl;

  // fill ExecTime:
  bool is_quantized = exec->graph().operands().at(node.getInputs().at(0)).typeInfo().type() ==
                      ir::DataType::QUANT_UINT8_ASYMM;

  uint32_t size = 0;
  for (const auto &ind : (node.getInputs() + node.getOutputs()) | ir::Remove::UNDEFINED)
  {
    size += exec->graph().operands().at(ind).info().total_size();
  }
  if (node_name == "Permute")
  {
    // TODO Change it to updateOperationExecTime()
    _et->updatePermuteTime(backend, backend, is_quantized, size, timer_res);
  }
  else
  {
    _et->updateOperationExecTime(backend, node_name, is_quantized, size, timer_res);
  }
};

TracingObserver::TracingObserver(const std::string &filepath, const ir::Graph &graph,
                                 const util::TracingCtx *tracing_ctx)
  : _recorder{std::make_unique<EventRecorder>()}, _collector{_recorder.get()}, _graph{graph},
    _tracing_ctx{tracing_ctx}
{
  _event_writer = EventWriter::get(filepath);
  _event_writer->startToUse();
}

TracingObserver::~TracingObserver()
{
  try
  {
    _event_writer->readyToFlush(std::move(_recorder));
  }
  catch (const std::exception &e)
  {
    std::cerr << "E: Fail to record event in TracingObserver: " << e.what() << std::endl;
  }
}

void TracingObserver::handleSubgraphBegin(ir::SubgraphIndex subg_ind)
{
  _collector.onEvent(
    EventCollector::SubgEvent{_tracing_ctx, EventCollector::Edge::BEGIN, subg_ind.value()});
}

void TracingObserver::handleJobBegin(IExecutor *, ir::SubgraphIndex subg_ind,
                                     ir::OperationIndex op_ind, const backend::Backend *backend)
{
  std::string backend_id = backend->config()->id();
  const auto &op = _graph.operations().at(op_ind);
  auto ev = EventCollector::OpSeqEvent{_tracing_ctx,     EventCollector::Edge::BEGIN,
                                       subg_ind.value(), backend_id,
                                       op_ind.value(),   op.name()};
  // add shape of inputs
  setUserData(_graph, &op, ev.userData);
  _collector.onEvent(ev);
}

void TracingObserver::handleJobEnd(IExecutor *, ir::SubgraphIndex subg_ind,
                                   ir::OperationIndex op_ind, const backend::Backend *backend)
{
  std::string backend_id = backend->config()->id();
  _collector.onEvent(EventCollector::OpSeqEvent{_tracing_ctx, EventCollector::Edge::END,
                                                subg_ind.value(), backend_id, op_ind.value(),
                                                _graph.operations().at(op_ind).name()});
}

void TracingObserver::handleSubgraphEnd(ir::SubgraphIndex subg_ind)
{
  _collector.onEvent(
    EventCollector::SubgEvent{_tracing_ctx, EventCollector::Edge::END, subg_ind.value()});
}

} // namespace exec

} // namespace onert
