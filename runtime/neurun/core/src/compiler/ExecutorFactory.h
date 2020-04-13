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

#ifndef __NEURUN_COMPILER_EXECUTOR_FACTORY_H__
#define __NEURUN_COMPILER_EXECUTOR_FACTORY_H__

#include <unordered_map>

#include "exec/IExecutor.h"
#include "ir/LoweredGraph.h"

namespace neurun
{
namespace compiler
{

class ExecutorFactory
{
public:
  static ExecutorFactory &get();

public:
  exec::IExecutor *create(std::unique_ptr<ir::LoweredGraph> lowered_graph,
                          const compiler::CompilerOptions &options);

private:
  ExecutorFactory();

private:
  static void initializeBackendContext(ir::LoweredGraph *lowered_graph);
  static exec::IExecutor *createLinearExecutor(std::unique_ptr<ir::LoweredGraph> lowered_graph,
                                               const compiler::CompilerOptions &options);
  static exec::IExecutor *createDataflowExecutor(std::unique_ptr<ir::LoweredGraph> lowered_graph,
                                                 const compiler::CompilerOptions &options,
                                                 bool parallel);

private:
  std::unordered_map<std::string,
                     std::function<exec::IExecutor *(std::unique_ptr<ir::LoweredGraph>,
                                                     const compiler::CompilerOptions &options)>>
      _map;
};

} // namespace compiler
} // namespace neurun

#endif // __NEURUN_COMPILER_EXECUTOR_FACTORY_H__
