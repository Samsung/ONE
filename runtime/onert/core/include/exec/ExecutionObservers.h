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
#include "ir/OpSequence.h"
#include "ExecTime.h"
#include "util/ITimer.h"
#include "IExecutor.h"
#include "misc/EventCollector.h"
#include "misc/EventRecorder.h"

namespace onert
{
namespace exec
{
class IExecutionObserver
{
public:
  /// @brief Invoked just before model (not individual operation) execution begins
  virtual void handleBegin(IExecutor *) { return; }

  virtual void handleBegin(IExecutor *, const ir::OpSequence *, const backend::Backend *) = 0;
  virtual void handleEnd(IExecutor *, const ir::OpSequence *, const backend::Backend *) = 0;

  /// @brief Invoked just after model (not individual operation) execution ends
  virtual void handleEnd(IExecutor *) { return; }

  virtual ~IExecutionObserver() = default;
};

class ProfileObserver : public IExecutionObserver
{
public:
  explicit ProfileObserver(std::shared_ptr<ExecTime> et) : _et(std::move(et)) {}
  void handleBegin(IExecutor *, const ir::OpSequence *, const backend::Backend *) override;
  void handleEnd(IExecutor *, const ir::OpSequence *, const backend::Backend *) override;

  void handleEnd(IExecutor *) override { _et->uploadOperationsExecTime(); }

private:
  std::unique_ptr<util::ITimer> _timer;
  std::shared_ptr<ExecTime> _et;
};

class ChromeTracingObserver : public IExecutionObserver
{
public:
  ChromeTracingObserver(const std::string &filepath);
  ~ChromeTracingObserver();
  void handleBegin(IExecutor *) override;
  void handleBegin(IExecutor *, const ir::OpSequence *, const backend::Backend *) override;
  void handleEnd(IExecutor *, const ir::OpSequence *, const backend::Backend *) override;
  void handleEnd(IExecutor *) override;

private:
  static std::string opSequenceTag(const ir::OpSequence *op_seq);

private:
  std::ofstream _ofs;
  EventRecorder _recorder;
  EventCollector _collector;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_OBSREVERS_H__
