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

#include "exec/ExecutionObservers.h"

#include <string>
#include <sstream>

#include "util/logging.h"
#include "exec/IExecutor.h"
#include "misc/polymorphic_downcast.h"
#include "ir/OpSequence.h"
#include "util/EventWriter.h"
#include "util/Utils.h"

namespace
{

void setUserData(const onert::ir::Graph &g, const onert::ir::OpSequence *op_seq,
                 decltype(EventCollector::Event::userData) &data)
{
  if (op_seq->size() == 0)
    return;

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

  const auto &first_op_idx = op_seq->operations().at(0);
  const auto &first_op_node = g.operations().at(first_op_idx);

  auto &inputs = first_op_node.getInputs();
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
                                     const ir::OpSequence *, const onert::backend::Backend *backend)
{
  _timer = backend->config()->timer();
  if (_timer == nullptr)
    throw std::runtime_error("To profile backend timer() method must be implemented");
  _timer->handleBegin();
}

void ProfileObserver::handleJobEnd(IExecutor *exec, ir::SubgraphIndex, const ir::OpSequence *op_seq,
                                   const backend::Backend *backend)
{
  _timer->handleEnd();
  const auto timer_res = _timer->getTime();

  // NOTE This assumes there is just one operation in a op_seq
  const auto &node = _graph.operations().at(op_seq->operations().at(0));
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
  // TODO Remove below after using _tracing_ctx
  UNUSED_RELEASE(_tracing_ctx);

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
  // TODO Write subg_ind into profling result
  UNUSED_RELEASE(subg_ind);
  _collector.onEvent(EventCollector::Event{EventCollector::Edge::BEGIN, "runtime", "Graph"});
}

void TracingObserver::handleJobBegin(IExecutor *, ir::SubgraphIndex subg_ind,
                                     const ir::OpSequence *op_seq, const backend::Backend *backend)
{
  // TODO Write subg_ind into profling result
  UNUSED_RELEASE(subg_ind);

  std::string backend_id = backend->config()->id();

  auto ev = EventCollector::Event{EventCollector::Edge::BEGIN, backend_id,
                                  opSequenceTag(op_seq, _graph.operations())};
  // add shape of inputs
  setUserData(_graph, op_seq, ev.userData);

  _collector.onEvent(ev);
}

void TracingObserver::handleJobEnd(IExecutor *, ir::SubgraphIndex subg_ind,
                                   const ir::OpSequence *op_seq, const backend::Backend *backend)
{
  // TODO Write subg_ind into profling result
  UNUSED_RELEASE(subg_ind);

  std::string backend_id = backend->config()->id();
  _collector.onEvent(EventCollector::Event{EventCollector::Edge::END, backend_id,
                                           opSequenceTag(op_seq, _graph.operations())});
}

void TracingObserver::handleSubgraphEnd(ir::SubgraphIndex subg_ind)
{
  // TODO Write subg_ind into profling result
  UNUSED_RELEASE(subg_ind);

  _collector.onEvent(EventCollector::Event{EventCollector::Edge::END, "runtime", "Graph"});
}

std::string TracingObserver::opSequenceTag(const ir::OpSequence *op_seq,
                                           const ir::Operations &operations)
{
  if (op_seq->size() == 0)
    return "Empty OpSequence";

  const auto &first_op_idx = op_seq->operations().at(0);
  const auto &first_op_node = operations.at(first_op_idx);
  std::string tag = "$" + std::to_string(first_op_idx.value());
  tag += " " + first_op_node.name();
  if (op_seq->size() > 1)
  {
    tag += " (+" + std::to_string(op_seq->size() - 1) + ")";
  }
  return tag;
}

} // namespace exec

} // namespace onert
