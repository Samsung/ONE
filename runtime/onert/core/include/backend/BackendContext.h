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
#include "ir/LowerInfoMap.h"
#include "exec/FunctionSequence.h"

namespace onert
{
namespace backend
{

class Backend;
class IConstantInitializer;
class IKernelGenerator;
struct ITensorRegistry;

using FunctionMap =
    std::vector<std::pair<ir::OpSequenceIndex, std::unique_ptr<exec::FunctionSequence>>>;

class BackendContext
{
public:
  struct OperationInfo
  {
    ir::OperationIndex index;
    ir::Layout layout;

    OperationInfo(ir::OperationIndex index, ir::Layout layout) : index{index}, layout{layout} {}
  };

public:
  BackendContext(const Backend *backend, const ir::Graph *graph,
                 std::shared_ptr<ITensorRegistry> tensor_registry = nullptr,
                 std::shared_ptr<IConstantInitializer> constant_initializer = nullptr,
                 std::shared_ptr<IKernelGenerator> kernel_gen = nullptr)
      : _backend{backend}, _graph{graph}, tensor_registry{tensor_registry},
        constant_initializer{constant_initializer}, kernel_gen{kernel_gen}
  {
  }

  virtual ~BackendContext() = default;

  void initialize(const std::vector<OperationInfo> &operation_list,
                  const std::vector<ir::OperandIndex> &operand_list);
  void initConsts();

  const Backend *backend() const { return _backend; }
  const ir::Graph *graph() const { return _graph; }
  const std::vector<OperationInfo> &operation_list() const { return _operation_list; }
  const std::vector<ir::OperandIndex> &operand_list() const { return _operand_list; }

  virtual ITensorRegistry *genTensors(const std::vector<onert::ir::OpSequenceIndex> &,
                                      const ir::OpSequences &, const ir::LowerInfoMap &)
  {
    return nullptr;
  }
  virtual FunctionMap genKernels(const std::vector<onert::ir::OpSequenceIndex> &,
                                 const ir::OpSequences &)
  {
    return {};
  }

private:
  const Backend *_backend{nullptr};
  const ir::Graph *_graph{nullptr};
  std::vector<OperationInfo> _operation_list;
  std::vector<ir::OperandIndex> _operand_list;

public:
  std::shared_ptr<ITensorRegistry> tensor_registry;
  std::shared_ptr<IConstantInitializer> constant_initializer;
  std::shared_ptr<IKernelGenerator> kernel_gen;
};

using BackendContexts = std::unordered_map<const Backend *, std::unique_ptr<BackendContext>>;

} // namespace backend
} // namespace onert

#endif // __ONERT_BACKEND_BACKEND_CONTEXT_H__
