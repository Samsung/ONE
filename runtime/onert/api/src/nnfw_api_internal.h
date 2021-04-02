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

#ifndef __API_NNFW_API_INTERNAL_H__
#define __API_NNFW_API_INTERNAL_H__

#include "nnfw.h"
#include "nnfw_experimental.h"
#include "CustomKernelRegistry.h"

#include <util/GeneralConfigSource.h>
#include <util/TracingCtx.h>
#include <exec/Execution.h>
#include <ir/Subgraphs.h>
#include <compiler/Compiler.h>

#include <string>
#include <memory>

struct nnfw_session
{
public:
  /**
   * @brief Enum class to express the session's state
   *
   * State transition diagram:
   *
   *           +--------------+
   *           | INITIALIZED  |
   *           +--------------+
   *             |
   *             | load_model
   *             v
   *           +--------------+
   *           | MODEL_LOADED |
   *           +--------------+
   *             |
   *             | prepare
   *             v
   *           +--------------+
   *           |   PREPARED   | --------+
   *           +--------------+         |
   *             |                      |
   *             | run                  |
   *             v                      |
   *           +--------------+  run    |
   *           |              | -----+  |
   *   +-----> | FINISHED_RUN |      |  | run_async
   *   |       |              | <----+  |
   *   |       +--------------+         |
   *   |         |                      |
   *   | await   | run_async            |
   *   |         v                      |
   *   |       +--------------+         |
   *   +------ |   RUNNING    | <-------+
   *           +--------------+
   */
  enum class State
  {
    INITIALIZED,  //< Session is initialized and nothing has done to it
    MODEL_LOADED, //< Model is loaded
    PREPARED,     //< Prepared(compiled) for execution
    RUNNING,      //< Execution is in progress (only for asynchronous execution)
    FINISHED_RUN  //< Executed at least once
  };

public:
  nnfw_session();
  ~nnfw_session();

  NNFW_STATUS prepare();
  NNFW_STATUS run();

  NNFW_STATUS run_async();
  NNFW_STATUS await();

public:
  const onert::ir::Graph *primary_subgraph();
  bool isStateInitialized();
  bool isStateModelLoaded();
  bool isStatePrepared();
  bool isStateRunning();
  bool isStateFinishedRun();
  bool isStatePreparedOrFinishedRun();

public:
  State _state{State::INITIALIZED};
  std::shared_ptr<onert::ir::Subgraphs> _subgraphs;
  std::unique_ptr<onert::compiler::Compiler> _compiler;
  std::unique_ptr<onert::exec::Execution> _execution;
  std::shared_ptr<onert::api::CustomKernelRegistry> _kernel_registry;

  std::unique_ptr<onert::util::TracingCtx> _tracing_ctx;
};

#endif // __API_NNFW_API_INTERNAL_H__
