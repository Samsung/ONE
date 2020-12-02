/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __COMPILATION_H__
#define __COMPILATION_H__

#include "ANeuralNetworksModel.h"

#include "compiler/Compiler.h"
#include "ir/Graph.h"
#include "ir/Subgraphs.h"
#include "exec/IExecutor.h"
#include "util/TracingCtx.h"

struct ANeuralNetworksCompilation
{
public:
  ANeuralNetworksCompilation(const ANeuralNetworksModel *model) noexcept;

public:
  bool finish() noexcept;

  onert::compiler::State state(void) noexcept { return _compiler->state(); }
  void publish(std::shared_ptr<onert::exec::ExecutorMap> &executors) noexcept
  {
    executors = _executors;
  }

private:
  std::shared_ptr<onert::ir::Subgraphs> _subgraphs;
  // TODO Refine the ownership of TracingCtx
  // In case of nnfw API, nnfw_session has ownership of TracingCtx.
  // In case of nnapi, there is no concept of session and primary model might have the ownership
  // of TracingCtx.
  // Since we don't support multiple models yet with nnapi in ONE, let's implement this later
  // and let's make it work with one model for now.
  std::unique_ptr<onert::util::TracingCtx> _tracing_ctx;

  std::shared_ptr<onert::compiler::Compiler> _compiler;
  std::shared_ptr<onert::exec::ExecutorMap> _executors;
};

#endif
