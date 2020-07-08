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

#ifndef __ONERT_EXEC_EXECUTOR_BASE_H__
#define __ONERT_EXEC_EXECUTOR_BASE_H__

#include <mutex>

#include "IPermuteFunction.h"
#include "Source.h"
#include "exec/ExecutionObservers.h"
#include "Sink.h"
#include "ShapeConverter.h"
#include "exec/IExecutor.h"
#include "ir/LoweredGraph.h"
#include "ir/LowerInfoMap.h"
#include "backend/IConfig.h"
#include "backend/Backend.h"
#include "compiler/OperandContext.h"
#include "exec/ExecTime.h"
#include "exec/IFunction.h"
#include "backend/IDynamicTensorManager.h"
#include "backend/ITensorManager.h"
#include "backend/ITensorBuilder.h"
#include "exec/ExecutionObservee.h"
#include "compiler/TensorBuilders.h"
#include <list>

namespace onert
{
namespace exec
{

class ExecutorBase : public IExecutor
{
public:
  /**
   * @brief Construct a new ExecutorBase object
   * @param graph Graph object
   * @param tensor_builders Tensor builders that are currently used
   */
  ExecutorBase(std::unique_ptr<ir::LoweredGraph> &&lowered_graph,
               const compiler::TensorBuilders &tensor_builders);

  virtual ~ExecutorBase() = default;

  const ir::Graph &graph() final { return _graph; }

  /**
   * @brief Execute without IODescription
   *
   * @param src_tensor Tensor list that will be copied to input tensors of this
   * @param pre_fn The permutation function that copy from src_tensor to input tensors of this
   */
  void execute(const std::vector<std::shared_ptr<backend::ITensor>> &src_tensors,
               const std::shared_ptr<IPermuteFunction> &pre_fn);

  void execute(const IODescription &desc) final;

  // Used only in Dataflow and Parallel Executors
  void setIndexedRanks(std::shared_ptr<ir::OperationIndexMap<int64_t>> ranks) final
  {
    _indexed_ranks = std::move(ranks);
  };

  virtual void executeImpl(void) = 0;

  void addObserver(std::unique_ptr<IExecutionObserver> ref) { _subject.add(std::move(ref)); };

  const std::vector<std::shared_ptr<backend::ITensor>> &getInputTensors() const
  {
    return _input_tensors;
  }

  const std::vector<std::shared_ptr<backend::ITensor>> &getOutputTensors() const
  {
    return _output_tensors;
  }

  const DynAllocInfoMap &getInputsDynamicAllocInfo() const { return _input_to_dyn_alloc_info; }

private:
  std::unique_ptr<ISource> source(const ir::IOIndex &index, const ir::TypeInfo &type,
                                  const void *buffer, size_t length, ir::Layout io_layout);
  std::unique_ptr<ISink> sink(const ir::IOIndex &index, const ir::TypeInfo &type, void *buffer,
                              size_t length, ir::Layout io_layout);

  template <typename T>
  std::unique_ptr<ISource> source(const ir::IOIndex &index, const void *buffer, size_t length,
                                  ir::Layout io_layout)
  {
    const auto operand_index = _graph.getInputs().at(index);
    const auto &operand = _graph.operands().at(operand_index);

    const auto tensor = _input_tensors[index.value()];
    const auto tensor_layout = tensor->layout();

    if (((io_layout == ir::Layout::NHWC) && (tensor_layout == ir::Layout::NCHW)) ||
        ((io_layout == ir::Layout::NCHW) && (tensor_layout == ir::Layout::NHWC)))
    {
      return std::make_unique<PermutateSource<T>>(buffer, length, operand.shape(), io_layout);
    }
    // TODO Change this to return error
    assert(io_layout != ir::Layout::UNKNOWN ||
           (tensor_layout != ir::Layout::NCHW && tensor_layout != ir::Layout::NCHW));

    return std::make_unique<CopySource<T>>(buffer, length, operand.shape());
  }

  template <typename T>
  std::unique_ptr<ISink> sink(const ir::IOIndex &index, void *buffer, size_t length,
                              ir::Layout io_layout)
  {
    const auto operand_index = _graph.getOutputs().at(index);
    const auto &operand = _graph.operands().at(operand_index);
    const auto tensor = _output_tensors[index.value()];
    const auto tensor_layout = tensor->layout();
    const auto tensor_shape = convertShape(tensor->getShape(), tensor->layout(), io_layout);

    if (((tensor_layout == ir::Layout::NCHW) && (io_layout == ir::Layout::NHWC)) ||
        ((tensor_layout == ir::Layout::NHWC) && (io_layout == ir::Layout::NCHW)))
    {
      return std::make_unique<PermutateSink<T>>(buffer, length, tensor_shape, io_layout);
    }
    // TODO Change this to return error
    assert(io_layout != ir::Layout::UNKNOWN ||
           (tensor_layout != ir::Layout::NCHW && tensor_layout != ir::Layout::NCHW));

    return std::make_unique<CopySink<T>>(buffer, length, operand.shape());
  }

protected:
  ExecutionObservee _subject;
  std::shared_ptr<ir::OperationIndexMap<int64_t>> _indexed_ranks;
  std::unique_ptr<ir::LoweredGraph> _lowered_graph;
  const ir::Graph &_graph;
  std::vector<std::shared_ptr<backend::ITensor>> _input_tensors;
  std::vector<std::shared_ptr<backend::ITensor>> _output_tensors;
  DynAllocInfoMap _input_to_dyn_alloc_info;
  DynAllocInfoMap _output_to_dyn_alloc_info;
  backend::TensorManagerSet _tensor_mgrs;
  std::mutex _mutex;

private:
  void handleDynamicInputTensor(ir::IOIndex input_index, const IODescription &desc);
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTOR_BASE_H__
