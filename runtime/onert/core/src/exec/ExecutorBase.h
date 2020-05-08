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

#include "Source.h"
#include "exec/ExecutionObservers.h"
#include "Sink.h"
#include "exec/IExecutor.h"
#include "ir/LoweredGraph.h"
#include "ir/LowerInfoMap.h"
#include "backend/IConfig.h"
#include "backend/Backend.h"
#include "compiler/OperandContext.h"
#include "exec/ExecTime.h"
#include "exec/IFunction.h"
#include "backend/ITensorManager.h"
#include "backend/ITensorBuilder.h"
#include "exec/ExecutionObservee.h"
#include <list>

namespace onert
{
namespace exec
{

struct InputTensorInfo
{
  std::shared_ptr<backend::ITensor> tensor;

  InputTensorInfo(std::shared_ptr<backend::ITensor> &tensor) : tensor(tensor) {}

  virtual ~InputTensorInfo() = default;
};

/**
 * @brief Class to have input tensor info when backend supports static tensor but not dynamic tensor
 */
struct InputTensorInfoForStaticTensor : public InputTensorInfo
{
  InputTensorInfoForStaticTensor(std::shared_ptr<backend::ITensor> &tensor)
      : InputTensorInfo(tensor)
  {
  }
};

/**
 * @brief Class to have input tensor info when backend supports static and dynamic tensor
 */
struct InputTensorInfoForDynamicTensor : public InputTensorInfo
{
  /// @brief index of input tensor whose memory needs to be allocated at execution time
  ir::OperandIndex ind;

  /// @brief dynamic tensor manager that can allocate memory when input tensor is dynamic
  backend::IDynamicTensorManager *tensor_manager;

  InputTensorInfoForDynamicTensor(std::shared_ptr<backend::ITensor> &tensor, ir::OperandIndex ind,
                                  backend::IDynamicTensorManager *tensor_manager)
      : InputTensorInfo(tensor), ind(ind), tensor_manager(tensor_manager)
  {
  }
};

class ExecutorBase : public IExecutor
{
public:
  /**
   * @brief Construct a new ExecutorBase object
   * @param graph Graph object
   * @param tensor_builders Tensor builders that are currently used
   */
  ExecutorBase(std::unique_ptr<ir::LoweredGraph> &&lowered_graph,
               const backend::TensorBuilderSet &tensor_builders);

  virtual ~ExecutorBase() = default;

  const ir::Graph &graph() final { return _graph; }

  void execute(const IODescription &desc) final;

  // Used only in Dataflow and Parallel Executors
  void setIndexedRanks(std::shared_ptr<ir::OperationIndexMap<int64_t>> ranks) final
  {
    _indexed_ranks = std::move(ranks);
  };

  virtual void executeImpl(void) = 0;

  void addObserver(std::unique_ptr<IExecutionObserver> ref) { _subject.add(std::move(ref)); };

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

    const auto tensor = _input_info[index.value()]->tensor;
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

    if (((tensor_layout == ir::Layout::NCHW) && (io_layout == ir::Layout::NHWC)) ||
        ((tensor_layout == ir::Layout::NHWC) && (io_layout == ir::Layout::NCHW)))
    {
      return std::make_unique<PermutateSink<T>>(buffer, length, operand.shape(), io_layout);
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
  std::vector<std::unique_ptr<InputTensorInfo>> _input_info;
  std::vector<std::shared_ptr<backend::ITensor>> _output_tensors;
  backend::TensorManagerSet _tensor_mgrs;
  std::mutex _mutex;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTOR_BASE_H__
