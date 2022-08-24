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

namespace onert
{
namespace ir
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
    unsigned long long h1 = (std::get<0>(edge.from).value() << 24) |
                            (std::get<1>(edge.from).value() << 16) | std::get<2>(edge.from).value();
    unsigned long long h2 = (std::get<0>(edge.to).value() << 24) |
                            (std::get<1>(edge.to).value() << 16) | std::get<2>(edge.to).value();
    return h1 + h2;
  }
};

inline std::ostream &operator<<(std::ostream &o, const IODesc &od)
{
  o << std::get<0>(od).value() << ":" << std::get<1>(od).value() << ":" << std::get<2>(od).value();
  return o;
}

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
  std::shared_ptr<Model> primary_model() { return _models.at(onert::ir::ModelIndex{0}); }

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
  const IODesc &input(uint32_t index) const { return _pkg_inputs[index]; }
  /**
   * @brief Get pkg_input at index
   *
   * @param[in] index Index of pkg_input to be returned
   * @return IODesc at index
   */
  IODesc &input(uint32_t index) { return _pkg_inputs[index]; }

  const std::vector<IODesc> &inputs() { return _pkg_inputs; }

  /**
   * @brief Add input at the end
   *
   * @param[in] input Input IODesc to be pushed
   */
  void addInput(const IODesc &input) { _pkg_inputs.push_back(input); }

  /**
   * @brief Get pkg_output at index
   *
   * @param[in] index Index of pkg_output to be returned
   * @return IODesc at index
   */
  const IODesc &output(uint32_t index) const { return _pkg_outputs[index]; }
  /**
   * @brief Get pkg_output at index
   *
   * @param[in] index Index of pkg_output to be returned
   * @return IODesc at index
   */
  IODesc &output(uint32_t index) { return _pkg_outputs[index]; }

  const std::vector<IODesc> &outputs() { return _pkg_outputs; }

  /**
   * @brief Add output at the end
   *
   * @param[in] output Output IODesc to be pushed
   */
  void addOutput(const IODesc &output) { _pkg_outputs.push_back(output); }

  /**
   * @brief Add edge between models at the end
   *
   * @param[in] from from IODesc
   * @param[in] to   to IODesc
   */
  void addEdge(const IODesc &from, const IODesc &to)
  {
    std::cout << from << " -> " << to << std::endl;
    _model_edges.insert(ModelEdge{from, to});
  }

  const std::unordered_set<ModelEdge, ModelEdgeHash, ModelEdgeEqual> &edges()
  {
    return _model_edges;
  }

  // TODO: Add iterate() or getter for edges

private:
  std::unordered_map<ModelIndex, std::shared_ptr<Model>> _models;
  std::vector<IODesc> _pkg_inputs;
  std::vector<IODesc> _pkg_outputs;
  std::unordered_set<ModelEdge, ModelEdgeHash, ModelEdgeEqual> _model_edges;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_NNPKG_H__
