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

#include "util/logging.h"
#include "exec/IExecutor.h"
#include "misc/polymorphic_downcast.h"
#include "ir/OpSequence.h"

namespace onert
{

namespace exec
{

void ProfileObserver::handleBegin(onert::exec::IExecutor *, const ir::OpSequence *,
                                  const onert::backend::Backend *backend)
{
  _timer = backend->config()->timer();
  if (_timer == nullptr)
    throw std::runtime_error("To profile backend timer() method must be implemented");
  _timer->handleBegin();
}

void ProfileObserver::handleEnd(IExecutor *exec, const ir::OpSequence *op_seq,
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
  for (const auto &ind : node.getInputs() + node.getOutputs())
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

ChromeTracingObserver::ChromeTracingObserver(const std::string &filepath, const ir::Graph &graph)
    : _ofs{filepath, std::ofstream::out}, _recorder{}, _collector{&_recorder}, _graph{graph}
{
}

ChromeTracingObserver::~ChromeTracingObserver()
{
  try
  {
    _recorder.writeToFile(_ofs);
  }
  catch (const std::exception &e)
  {
    std::cerr << "E: Fail to record event in ChromeTracingObserver: " << e.what() << std::endl;
  }
}

void ChromeTracingObserver::handleBegin(IExecutor *)
{
  _collector.onEvent(EventCollector::Event{EventCollector::Edge::BEGIN, "runtime", "Graph"});
}

void ChromeTracingObserver::handleBegin(IExecutor *, const ir::OpSequence *op_seq,
                                        const backend::Backend *backend)
{
  std::string backend_id = backend->config()->id();
  _collector.onEvent(EventCollector::Event{EventCollector::Edge::BEGIN, backend_id,
                                           opSequenceTag(op_seq, _graph.operations())});
}

void ChromeTracingObserver::handleEnd(IExecutor *, const ir::OpSequence *op_seq,
                                      const backend::Backend *backend)
{
  std::string backend_id = backend->config()->id();
  _collector.onEvent(EventCollector::Event{EventCollector::Edge::END, backend_id,
                                           opSequenceTag(op_seq, _graph.operations())});
}

void ChromeTracingObserver::handleEnd(IExecutor *)
{
  _collector.onEvent(EventCollector::Event{EventCollector::Edge::END, "runtime", "Graph"});
}

std::string ChromeTracingObserver::opSequenceTag(const ir::OpSequence *op_seq,
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
