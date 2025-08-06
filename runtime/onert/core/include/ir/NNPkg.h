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

#ifndef __ONERT_IR_NNPKG_H__
#define __ONERT_IR_NNPKG_H__

#include <memory>
#include <unordered_set>
#include <vector>

#include "ir/Index.h"
#include "ir/Model.h"
#include "util/Set.h"

namespace onert::ir
{

using IODesc = std::tuple<ModelIndex, SubgraphIndex, IOIndex>;

struct ModelEdge
{
  IODesc from;
  IODesc to;
};

struct ModelEdgeEqual
{
  bool operator()(const onert::ir::ModelEdge &lhs, const onert::ir::ModelEdge &rhs) const
  {
    return lhs.from == rhs.from && lhs.to == rhs.to;
  }
};

struct ModelEdgeHash
{
  size_t operator()(const ::onert::ir::ModelEdge &edge) const noexcept
  {
    const auto &[from_model_index, from_sug_index, from_io_index] = edge.from;
    unsigned long long h1 =
      (from_model_index.value() << 24) | (from_sug_index.value() << 16) | from_io_index.value();

    const auto &[to_model_index, to_sug_index, to_io_index] = edge.to;
    unsigned long long h2 =
      (to_model_index.value() << 24) | (to_sug_index.value() << 16) | to_io_index.value();
    return h1 + h2;
  }
};

inline std::ostream &operator<<(std::ostream &o, const IODesc &od)
{
  const auto &[m, s, i] = od;
  o << m.value() << ":" << s.value() << ":" << i.value();
  return o;
}

using ModelEdgeSet = std::unordered_set<ir::ModelEdge, ir::ModelEdgeHash, ir::ModelEdgeEqual>;

/**
 * @brief Struct to gather model I/O information in multimodel NN package
 *        Model I/O will have role one of below
 *        - Package input/output
 *        - Edge's start/finish point between model
 */
struct ModelEdges
{
  std::vector<ir::IODesc> pkg_inputs;
  std::vector<ir::IODesc> pkg_outputs;
  ModelEdgeSet edges;
};

class NNPkg
{
public:
  NNPkg() = default;
  NNPkg(const NNPkg &obj) = default;
  NNPkg(NNPkg &&) = default;
  NNPkg &operator=(const NNPkg &) = default;
  NNPkg &operator=(NNPkg &&) = default;
  ~NNPkg() = default;

  NNPkg(std::shared_ptr<Model> model) { _models[ModelIndex{0}] = model; }
  std::shared_ptr<Model> primary_model() const { return _models.at(onert::ir::ModelIndex{0}); }

  /**
   * @brief Put model at index
   *
   * @param[in] model Model to be pushed
   * @param[in] index Index where Model is to be pushed
   */
  void push(ModelIndex index, const std::shared_ptr<Model> &model) { _models[index] = model; }

  /**
   * @brief Get the count of model
   *
   * @return the count of models
   */
  size_t model_count() const { return _models.size(); }

  /**
   * @brief Get model at index
   *
   * @param[in] index Index of the model to be returned
   * @return Model at index
   */
  const std::shared_ptr<Model> &model(const ModelIndex &index) const { return _models.at(index); }
  /**
   * @brief Get model at index
   *
   * @param[in] index Index of the model to be returned
   * @return Model at index
   */
  std::shared_ptr<Model> &model(const ModelIndex &index) { return _models.at(index); }

  /**
   * @brief Get pkg_input at index
   *
   * @param[in] index Index of pkg_input to be returned
   * @return IODesc at index
   */
  const IODesc &input(uint32_t index) const { return _edges.pkg_inputs[index]; }
  /**
   * @brief Get pkg_input at index
   *
   * @param[in] index Index of pkg_input to be returned
   * @return IODesc at index
   */
  IODesc &input(uint32_t index) { return _edges.pkg_inputs[index]; }
  /**
   * @brief Add input at the end
   *
   * @param[in] input Input IODesc to be pushed
   */
  void addInput(const IODesc &input) { _edges.pkg_inputs.push_back(input); }

  /**
   * @brief Get pkg_output at index
   *
   * @param[in] index Index of pkg_output to be returned
   * @return IODesc at index
   */
  const IODesc &output(uint32_t index) const { return _edges.pkg_outputs[index]; }
  /**
   * @brief Get pkg_output at index
   *
   * @param[in] index Index of pkg_output to be returned
   * @return IODesc at index
   */
  IODesc &output(uint32_t index) { return _edges.pkg_outputs[index]; }
  /**
   * @brief Add output at the end
   *
   * @param[in] output Output IODesc to be pushed
   */
  void addOutput(const IODesc &output) { _edges.pkg_outputs.push_back(output); }

  /**
   * @brief Add edge between models at the end
   *
   * @param[in] from from IODesc
   * @param[in] to   to IODesc
   */
  void addEdge(const IODesc &from, const IODesc &to)
  {
    std::cout << from << " -> " << to << std::endl;
    _edges.edges.insert(ModelEdge{from, to});
  }
  /**
   * @brief   Get model edge set
   * @return  Edge set reference
   */
  const ModelEdges &model_edges() { return _edges; }

  /**
   * @brief Verify NNPkg
   *
   */
  void verify(void)
  {
    // Verify edges information
    //
    // Only duplicates of nnpkg output and Edge `from` are possible.
    // | Whether duplicates are possible   | Edge `to` | Edge `from` |
    // | nnpkg input  (input of subgraph)  | X (*1)    | X (*2)      |
    // | nnpkg output (output of subgraph) | X (*2)    | O           |
    // *1. The subjects who determine values of each buffer are different.
    //    - nnpkg input : user input
    //    - Edge `to`   : output of another subgraph
    // *2. `IOIndex` of inputs and outputs of subgraph is distinct.
    //
    for (const auto &edge : _edges.edges)
    {
      if (std::find(_edges.pkg_inputs.begin(), _edges.pkg_inputs.end(), edge.to) !=
          _edges.pkg_inputs.end())
      {
        throw std::runtime_error{
          "Invalid edge information. NNPkg inputs and Edge `to` cannot be duplicated"};
      }
    }
  }

  // TODO Find better way to handle single model NNPackage and multi model NNPackage on inputSize(),
  //      outputSize(), inputInfo(), outputInfo()

  /**
   * @brief   Get model input size
   */
  uint32_t inputSize() const
  {
    return _models.size() == 1 ? primary_model()->primary_subgraph()->getInputs().size()
                               : _edges.pkg_inputs.size();
  }

  /**
   * @brief   Get model output size
   */
  uint32_t outputSize() const
  {
    return _models.size() == 1 ? primary_model()->primary_subgraph()->getOutputs().size()
                               : _edges.pkg_outputs.size();
  }

  /**
   * @brief   Get model input info
   */
  const OperandInfo &inputInfo(uint32_t index) const
  {
    if (_models.size() == 1)
    {
      auto const graph = primary_model()->primary_subgraph();
      auto const operand_index = graph->getInputs().at(index);
      return graph->operands().at(operand_index).info();
    }

    auto const &desc = input(index);
    auto const graph = model(std::get<ModelIndex>(desc))->primary_subgraph();
    auto const operand_index = graph->getInputs().at(std::get<IOIndex>(desc).value());
    return graph->operands().at(operand_index).info();
  }

  /**
   * @brief   Get model output info
   */
  const OperandInfo &outputInfo(uint32_t index) const
  {
    if (_models.size() == 1)
    {
      auto const graph = primary_model()->primary_subgraph();
      auto const operand_index = graph->getOutputs().at(index);
      return graph->operands().at(operand_index).info();
    }

    auto const &desc = output(index);
    auto const graph = model(std::get<ModelIndex>(desc))->primary_subgraph();
    auto const operand_index = graph->getOutputs().at(std::get<IOIndex>(desc).value());
    return graph->operands().at(operand_index).info();
  }

  void changeInputShape(uint32_t index, const ir::Shape &new_shape)
  {
    if (_models.size() == 1)
    {
      auto graph = primary_model()->primary_subgraph();
      auto const operand_index = graph->getInputs().at(index);
      graph->changeShape(operand_index, new_shape);
      return;
    }

    auto const &desc = input(index);
    auto graph = model(std::get<ModelIndex>(desc))->primary_subgraph();
    auto const operand_index = graph->getInputs().at(std::get<IOIndex>(desc).value());
    graph->changeShape(operand_index, new_shape);
  }

  /**
   * @brief Replace model
   *
   * @param[in] model Model to be replaced
   *
   * TODO:  Support multiple models
   */
  void replaceModel(std::shared_ptr<Model> model) { _models[ModelIndex{0}] = model; }

  /**
   * @brief Reset models to empty state (no model)
   *        This is useful when we want to reduce the memory usage
   */
  void resetModels() { _models.clear(); }

  /**
   * @brief     Get the Package Outputs set object
   * @param[in] model_idx Model index
   * @param[in] subg_idx  Subgraph index
   * @return    Set of OperandIndex which are outputs of the given model and subgraph index
   */
  util::Set<ir::OperandIndex> getPkgOutputs(const ir::ModelIndex &model_idx,
                                            const ir::SubgraphIndex &subg_idx) const
  {
    util::Set<ir::OperandIndex> ret;

    // Assume that all outputs are from primary subgraph of each model
    if (subg_idx != ir::SubgraphIndex{0})
      return ret;

    if (model_count() == 1)
    {
      assert(model_idx == ir::ModelIndex{0});
      for (const auto &output : primary_model()->primary_subgraph()->getOutputs() |
                                  ir::Remove::UNDEFINED | ir::Remove::DUPLICATED)
        ret.add(output);

      return ret;
    }

    for (const auto &output_desc : _edges.pkg_outputs)
    {
      if (const auto &[m, s, io] = output_desc; m == model_idx && s == subg_idx)
      {
        auto idx = model(m)->primary_subgraph()->getOutputs().at(io.value());
        ret.add(idx);
      }
    }

    return ret;
  }

  // TODO: Add iterate() or getter for edges

private:
  std::unordered_map<ModelIndex, std::shared_ptr<Model>> _models;
  ModelEdges _edges;
};

} // namespace onert::ir

namespace std
{

template <> struct hash<onert::ir::IODesc>
{
  size_t operator()(const ::onert::ir::IODesc &iodesc) const noexcept
  {
    auto [m, s, i] = iodesc;
    return (m.value() << 24) | (s.value() << 16) | i.value();
  }
};

} // namespace std

#endif // __ONERT_IR_NNPKG_H__
