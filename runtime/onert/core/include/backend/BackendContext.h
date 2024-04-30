/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_BACKEND_BACKEND_CONTEXT_H__
#define __ONERT_BACKEND_BACKEND_CONTEXT_H__

#include <memory>
#include "ir/Graph.h"
#include "ir/OperationIndexMap.h"
#include "ir/OperandIndexMap.h"
#include "compiler/GraphLowerInfo.h"
#include "exec/FunctionSequence.h"

namespace onert
{
namespace backend
{

class Backend;
struct ITensorRegistry;

using FunctionMap = std::unordered_map<ir::OperationIndex, std::unique_ptr<exec::FunctionSequence>>;

struct ContextData
{
  /* A partial graph that only includes used operand/operations of the original graph */
  std::unique_ptr<ir::Graph> graph;
  /* A linear order of operations. This is neccessary for when a graph is not fully connected */
  std::vector<onert::ir::OperationIndex> op_order;
  /* Operands that are defined by other backends */
  util::Set<ir::OperandIndex> external_operands;
  /* Operand layout info */
  ir::OperandIndexMap<ir::Layout> operand_layouts;
  /* Custom kernel builder */
  std::shared_ptr<custom::IKernelBuilder> custom_kernel_builder;
  /* Is linear executor or not */
  bool is_linear_executor;
};

class BackendContext
{
public:
  BackendContext(const Backend *backend, ContextData &&data,
                 std::shared_ptr<ITensorRegistry> tensor_registry = nullptr)
    : _backend{backend}, _data{std::move(data)}, tensor_registry{tensor_registry}
  {
  }

  virtual ~BackendContext() = default;

  const Backend *backend() const { return _backend; }
  const ir::Graph *graph() const { return _data.graph.get(); }
  const util::Set<ir::OperandIndex> &external_operands() const { return _data.external_operands; }
  const ir::OperandIndexMap<ir::Layout> &operand_layouts() const { return _data.operand_layouts; }
  const ContextData &data() const { return _data; }

  virtual ITensorRegistry *genTensors() = 0;
  virtual FunctionMap genKernels() = 0;

protected:
  const Backend *_backend{nullptr};
  ContextData _data;

public:
  std::shared_ptr<ITensorRegistry> tensor_registry;
};

using BackendContexts = std::unordered_map<const Backend *, std::unique_ptr<BackendContext>>;

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BACKEND_CONTEXT_H__
