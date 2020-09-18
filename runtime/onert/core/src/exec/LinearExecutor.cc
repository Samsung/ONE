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

#include "LinearExecutor.h"
#ifdef RUY_PROFILER
#include "ruy/profiler/instrumentation.h"
#endif

namespace onert
{
namespace exec
{

#ifdef RUY_PROFILER
namespace
{
char *seq_to_label(const onert::ir::OpSequence *op_seq, const onert::ir::Operations &operations)
{
  auto node_name = operations.at(*op_seq->begin()).name();
  char *cstr = new char[node_name.length() + 1];
  std::strcpy(cstr, node_name.c_str());
  return cstr;
}
} // namespace
#endif

void LinearExecutor::executeImpl()
{
  _subject.notifyModelBegin(this);
  int idx = 0;
  for (auto &&code : _code)
  {
    const auto op_seq = code.op_seq;
    const auto backend = code.lower_info->backend();
// TODO : Move ruy profiler into ExecutionObserver
#ifdef RUY_PROFILER
    ruy::profiler::ScopeLabel label(seq_to_label(op_seq, _graph.operations()));
#endif
    _subject.notifyJobBegin(this, op_seq, backend);

    auto &fn_seq = code.fn_seq;
    fn_seq->initRunning();

    bool handle_dynamic_tensor = op_seq->has_dynamic_tensor() || hasDynamicInput();
    VERBOSE_F() << "FUNCTION SEQ RUNNING " << idx++ << std::endl;
    fn_seq->enableDynamicShapeInferer(handle_dynamic_tensor);
    fn_seq->run();

    _subject.notifyJobEnd(this, op_seq, backend);
  }
  _subject.notifyModelEnd(this);
}

} // namespace exec
} // namespace onert
