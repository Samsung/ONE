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

#ifndef __ONERT_COMPILER_EXECUTOR_FACTORY_H__
#define __ONERT_COMPILER_EXECUTOR_FACTORY_H__

#include <unordered_map>

#include "backend/ITensor.h"
#include "exec/IExecutor.h"
#include "ir/LoweredGraph.h"
#include "TensorBuilders.h"

namespace onert
{
namespace compiler
{

class ExecutorFactory
{
public:
  static ExecutorFactory &get();

public:
  exec::IExecutor *create(std::unique_ptr<ir::LoweredGraph> lowered_graph,
                          const compiler::CompilerOptions &options,
                          const std::shared_ptr<exec::ExecutorMap> &executor_map);

private:
  ExecutorFactory();

private:
  static void initializeBackendContext(ir::LoweredGraph *lowered_graph);
  static void runTensorRegistration(ir::LoweredGraph *lowered_graph,
                                    const std::vector<ir::OpSequenceIndex> &order);
  static std::vector<std::shared_ptr<backend::ITensor>>
  initializeModelIOTensors(ir::LoweredGraph &lowered_graph,
                           const ir::OperandIndexSequence &indices);
  static void prepareExternalTensors(ir::LoweredGraph &lowered_graph,
                                     TensorBuilders &tensor_builders);
  static exec::IExecutor *
  createLinearExecutor(std::unique_ptr<ir::LoweredGraph> lowered_graph,
                       const compiler::CompilerOptions &options,
                       const std::shared_ptr<exec::ExecutorMap> &executor_map);
  static exec::IExecutor *
  createDataflowExecutor(std::unique_ptr<ir::LoweredGraph> lowered_graph,
                         const compiler::CompilerOptions &options,
                         const std::shared_ptr<exec::ExecutorMap> &executor_map, bool parallel);

private:
  std::unordered_map<
      std::string, std::function<exec::IExecutor *(
                       std::unique_ptr<ir::LoweredGraph>, const compiler::CompilerOptions &options,
                       const std::shared_ptr<exec::ExecutorMap> &executor_map)>>
      _map;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_EXECUTOR_FACTORY_H__
