/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_EXEC_EXECUTORS_H__
#define __ONERT_EXEC_EXECUTORS_H__

#include "exec/IExecutors.h"
#include "ir/NNPkg.h"
#include "IPermuteFunction.h"

namespace std
{

template <> struct hash<std::pair<::onert::ir::ModelIndex, ::onert::ir::SubgraphIndex>>
{
  size_t operator()(
    const std::pair<::onert::ir::ModelIndex, ::onert::ir::SubgraphIndex> &pair) const noexcept
  {
    return (hash<uint32_t>()(pair.first.value()) << 16) ^ hash<uint32_t>()(pair.second.value());
  }
};

} // namespace std

namespace onert
{
namespace exec
{

/**
 * @brief Class to gather executors
 */
class MultiModelExecutors : public IExecutors
{
public:
  MultiModelExecutors(void) = delete;
  MultiModelExecutors(std::unique_ptr<ir::ModelEdges> model_edges)
    : _executors{}, _model_edges{std::move(model_edges)}, _edge_quant_layers{},
      _edge_quant_tensors{}, _edge_tensors{}, _is_created_edge_quant_layers{false},
      _pkg_input_quant_layers{}, _pkg_output_quant_layers{}, _pkg_input_quant_tensors{},
      _pkg_output_quant_tensors{}, _pkg_input_tensors{}, _pkg_output_tensors{}
  {
    for (const auto &edge : _model_edges->edges)
    {
      _edge_map[edge.from].emplace_back(edge.to);
    }
  }
  MultiModelExecutors(const MultiModelExecutors &) = delete;
  MultiModelExecutors(MultiModelExecutors &&) = default;
  ~MultiModelExecutors() = default;

  // TODO Use Executor index
  void emplace(const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index,
               std::unique_ptr<IExecutor> exec) override;

  IExecutor *at(const ir::ModelIndex &model_index,
                const ir::SubgraphIndex &subg_index) const override;

  uint32_t inputSize() const override;

  uint32_t outputSize() const override;

  const ir::OperandInfo &inputInfo(const ir::IOIndex &index) const override;

  const ir::OperandInfo &outputInfo(const ir::IOIndex &index) const override;

  void execute(const ExecutionContext &ctx) override;

private:
  void checkSupportedMultimodel() const;
  void createEdgeQuantLayers();
  void CreatePkgIOTensors(const IODescription &desc);
  void createPkgIOQuantLayers(const IODescription &desc);
  uint16_t modelCount() const;

private:
  // TODO Remove this class
  class PermuteLayer : public exec::IPermuteFunction
  {
  public:
    PermuteLayer(const std::vector<backend::ITensor *> &inputs,
                 const std::vector<backend::ITensor *> &outputs)
    {
      assert(inputs.size() == outputs.size());
      _src_tensors = inputs;
      _dst_tensors = outputs;
    }
    virtual ~PermuteLayer() {}
    void optimize() override {}
  };

  class EdgeTensor;

private:
  std::unordered_map<std::pair<ir::ModelIndex, ir::SubgraphIndex>, std::unique_ptr<IExecutor>>
    _executors;

  // NOTE _model_edges may use different struct type for executor implementation
  std::unique_ptr<ir::ModelEdges> _model_edges;
  std::unordered_map<ir::IODesc, std::vector<ir::IODesc>> _edge_map;

  /**
   * @brief Type-aware quantization layers for edges between executors
   *
   */
  // TODO Move variables related to type-aware quantization for edges into compilation stage
  // TODO Replace PermuteLayer with backend::builtin::kernel::PermuteLayer
  std::unordered_map<std::pair<ir::ModelIndex, ir::SubgraphIndex>, std::unique_ptr<PermuteLayer>>
    _edge_quant_layers;

  /**
   * @brief Tensors for type-aware quantization of edges
   *        Key: `to` IODesc, Value: EdgeTensor
   */
  //
  // Q: Why is Key `to` IODesc
  // A: these tensors are currently created depending on the type of `to`
  // TODO Unify tensors with the same `from` tensor and same type
  // NOTE The incomplete type 'EdgeTensor' cannot be declared as unique_ptr.
  std::unordered_map<ir::IODesc, std::shared_ptr<EdgeTensor>> _edge_quant_tensors;

  /**
   * @brief Tensors for edges between executors that are not related to type-aware quantization
   *        Key: `from` IODesc, Value: EdgeTensor
   */
  // Q: Why is Key `from` IODesc
  // A: `from` can be connected to multiple `to`
  // NOTE The incomplete type 'EdgeTensor' cannot be declared as unique_ptr.
  std::unordered_map<ir::IODesc, std::shared_ptr<EdgeTensor>> _edge_tensors;
  /**
   * @brief Whether type-aware quantization layers for edges between executors are created
   *
   */
  // TODO Remove this member after the creation of type-aware quantization layers for edges
  //      is moved into compilation stage
  bool _is_created_edge_quant_layers;

  // TODO Replace PermuteLayer with backend::builtin::kernel::PermuteLayer
  std::unordered_map<std::pair<ir::ModelIndex, ir::SubgraphIndex>, std::unique_ptr<PermuteLayer>>
    _pkg_input_quant_layers;
  // TODO Replace PermuteLayer with backend::builtin::kernel::PermuteLayer
  std::unordered_map<std::pair<ir::ModelIndex, ir::SubgraphIndex>, std::unique_ptr<PermuteLayer>>
    _pkg_output_quant_layers;
  // Edge tensors of nnpkg inputs/outputs for type-aware quantization
  std::unordered_map<ir::IODesc, std::shared_ptr<EdgeTensor>> _pkg_input_quant_tensors;
  std::unordered_map<ir::IODesc, std::shared_ptr<EdgeTensor>> _pkg_output_quant_tensors;
  // IOTensors for user buffer
  std::unordered_map<ir::IODesc, std::unique_ptr<backend::builtin::IOTensor>> _pkg_input_tensors;
  std::unordered_map<ir::IODesc, std::unique_ptr<backend::builtin::IOTensor>> _pkg_output_tensors;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTORS_H__
