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
  size_t
  operator()(const std::pair<::onert::ir::ModelIndex, ::onert::ir::SubgraphIndex> &pair) const
    noexcept
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
class Executors : public IExecutors
{
public:
  Executors(void) = delete;
  Executors(std::unique_ptr<ir::ModelEdges> model_edges)
    : _executors{}, _model_edges{std::move(model_edges)}, _edge_quant_layers{},
      _edge_quant_tensors{}, _edge_tensors{}, _is_created_edge_quant_layers{false},
      _pkg_input_quant_layers{}, _pkg_output_quant_layers{}, _pkg_io_edge_tensors{}, _pkg_inputs{},
      _pkg_outputs{}
  {
    for (const auto &edge : _model_edges->edges)
    {
      _edge_map[edge.from].emplace_back(edge.to);
    }
  }
  Executors(const Executors &) = delete;
  Executors(Executors &&) = default;
  ~Executors() = default;

  // TODO Use Executor index
  void emplace(const ir::ModelIndex &model_index, const ir::SubgraphIndex &subg_index,
               std::unique_ptr<IExecutor> exec) override;

  IExecutor *at(const ir::ModelIndex &model_index,
                const ir::SubgraphIndex &subg_index) const override;

  uint32_t inputSize() const override;

  uint32_t outputSize() const override;

  const ir::OperandInfo &inputInfo(const ir::IOIndex &index) const override;

  const ir::OperandInfo &outputInfo(const ir::IOIndex &index) const override;

  void execute(const IODescription &desc) override;

private:
  void checkSupportedMultimodel() const;
  void createEdgeQuantLayers();
  void InitPkgIO(const IODescription &desc);
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

  // TODO Move variables related to type-aware quantization for edges into compilation stage
  // TODO Replace PermuteLayer with backend::builtin::kernel::PermuteLayer
  std::unordered_map<std::pair<ir::ModelIndex, ir::SubgraphIndex>, std::unique_ptr<PermuteLayer>>
    _edge_quant_layers;
  // TODO Unify tensors with the same `from` tensor and same type
  // NOTE The incomplete type 'EdgeTensor' cannot be declared as unique_ptr.
  std::unordered_map<ir::IODesc, std::shared_ptr<EdgeTensor>> _edge_quant_tensors;
  // Key: `from` IODesc, Value: EdgeTensor
  // NOTE The incomplete type 'EdgeTensor' cannot be declared as unique_ptr.
  std::unordered_map<ir::IODesc, std::shared_ptr<EdgeTensor>> _edge_tensors;
  bool _is_created_edge_quant_layers;

  // TODO Replace PermuteLayer with backend::builtin::kernel::PermuteLayer
  std::unordered_map<std::pair<ir::ModelIndex, ir::SubgraphIndex>, std::unique_ptr<PermuteLayer>>
    _pkg_input_quant_layers;
  // TODO Replace PermuteLayer with backend::builtin::kernel::PermuteLayer
  std::unordered_map<std::pair<ir::ModelIndex, ir::SubgraphIndex>, std::unique_ptr<PermuteLayer>>
    _pkg_output_quant_layers;
  // Assumption: nnpkg inputs/outputs are not duplicated.
  std::unordered_map<ir::IODesc, std::shared_ptr<EdgeTensor>> _pkg_io_edge_tensors;
  std::vector<std::unique_ptr<backend::builtin::IOTensor>> _pkg_inputs;
  std::vector<std::unique_ptr<backend::builtin::IOTensor>> _pkg_outputs;
};

} // namespace exec
} // namespace onert

#endif // __ONERT_EXEC_EXECUTORS_H__
