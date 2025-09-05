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

#ifndef __ONERT_IR_MODEL_H__
#define __ONERT_IR_MODEL_H__

#include "ir/IGraph.h"
#include "ir/Index.h"
#include "util/ObjectManager.h"

#include <memory>
#include <unordered_map>

namespace onert::backend::custom
{
class IKernelBuilder;
} // namespace onert::backend::custom

namespace onert::ir
{

class Model
{
public:
  Model() = default;
  Model(const Model &obj) = default;
  Model(Model &&) = default;
  Model &operator=(const Model &) = default;
  Model &operator=(Model &&) = default;
  ~Model() = default;

  /**
   * @brief Put subgraph in the container with a new Index for that
   *
   * @param[in] subg Subgraph to be pushed
   * @param[in] index Index of subgraph to be pushed
   * @return Created
   */
  void push(SubgraphIndex index, const std::shared_ptr<IGraph> &subg) { _subgraphs[index] = subg; }

  /**
   * @brief Remove the subgraph that is associated with the given index
   *
   * @param[in] index Index of the subgraph to be removed
   * @return N/A
   */
  void remove(const SubgraphIndex &index) { _subgraphs.erase(index); }

  /**
   * @brief Get the subgraph that is associated with the given index
   *
   * @param[in] index Index of the subgraph to be returned
   * @return IGraph
   */
  const std::shared_ptr<IGraph> &at(const SubgraphIndex &index) const
  {
    return _subgraphs.at(index);
  }
  /**
   * @brief Get the subgraph that is associated with the given index
   *
   * @param[in] index Index of the subgraph to be returned
   * @return IGraph
   */
  std::shared_ptr<IGraph> &at(const SubgraphIndex &index) { return _subgraphs.at(index); }

  /**
   * @brief Get the subgraph that is associated with the given index
   *
   * @param[in] index Index of the subgraph to be returned
   * @return true if such entry exists otherwise false
   */
  bool exist(const SubgraphIndex &index) const
  {
    auto it = _subgraphs.find(index);
    return it != _subgraphs.end();
  }

  /**
   * @brief Iterate over the container with given function
   *
   * @param[in] fn Function to be run for every container entry
   * @return N/A
   */
  void iterate(const std::function<void(const SubgraphIndex &, const IGraph &)> &fn) const
  {
    for (const auto &[subg_idx, subg] : _subgraphs)
    {
      fn(subg_idx, *subg);
    }
  }

  /**
   * @brief Iterate over the container with given function
   *
   * @param[in] fn Function to be run for every container entry
   * @return N/A
   */
  void iterate(const std::function<void(const SubgraphIndex &, IGraph &)> &fn)
  {
    for (const auto &[subg_idx, subg] : _subgraphs)
    {
      fn(subg_idx, *subg);
    }
  }

  /**
   * @brief Get count of Subgraphs
   *
   * @return count of Subgraphs
   */
  size_t subgraphs_count() const { return _subgraphs.size(); }

  /**
   * @brief Return the primary subgraph
   *
   * @return std::shared_ptr<IGraph> Primary subgraph
   */
  std::shared_ptr<IGraph> primary_subgraph() const { return _subgraphs.at(SubgraphIndex{0}); }

  /**
   * @brief Return whether the model has only typename Graph
   *
   * @tparam Graph Type that inherits from IGraph
   *
   * @return true if the model has only typename Graph, otherwise false
   */
  template <typename Graph, std::enable_if_t<std::is_base_of_v<IGraph, Graph>, bool> = true>
  bool hasOnly()
  {
    for (const auto &e : _subgraphs)
    {
      if (std::dynamic_pointer_cast<Graph>(e.second) == nullptr)
        return false;
    }
    return true;
  }

private:
  std::unordered_map<SubgraphIndex, std::shared_ptr<IGraph>> _subgraphs;

  // Custom operations support
public:
  void
  bindKernelBuilder(const std::shared_ptr<onert::backend::custom::IKernelBuilder> &kernel_builder)
  {
    _kernel_builder = kernel_builder;
  }

  const std::shared_ptr<backend::custom::IKernelBuilder> &getKernelBuilder() const
  {
    return _kernel_builder;
  }

private:
  std::shared_ptr<backend::custom::IKernelBuilder> _kernel_builder;

public:
  void add_metadata(const std::string &name, std::unique_ptr<const ir::Data> data)
  {
    _metadatas.emplace(name, std::move(data));
  }

  bool exists_metadata(const std::string &name) const
  {
    return _metadatas.find(name) != _metadatas.end();
  }

  // NOTE The corresponding metadata is deleted from the model and returned
  std::unique_ptr<const ir::Data> extract_metadata(const std::string name)
  {
    auto m = _metadatas.find(name);

    if (m == _metadatas.end())
      throw std::out_of_range{"no meatdata named " + name};

    auto data = std::move(m->second);
    _metadatas.erase(m);
    return data;
  }

  void addSignatureMap(const ir::SubgraphIndex &index, const std::string &name)
  {
    _signature_map[index] = name;
  }

  const std::unordered_map<ir::SubgraphIndex, std::string> &signatureMap() const
  {
    return _signature_map;
  }

private:
  // TODO: Apply Heterogeneous lookup for unordered containers (transparent hashing) since C++20
  //       to use `std::string_view` with lookup functions in unordered containers
  std::unordered_map<std::string, std::unique_ptr<const ir::Data>> _metadatas;
  std::unordered_map<ir::SubgraphIndex, std::string> _signature_map;
};
} // namespace onert::ir

#endif // __ONERT_IR_MODEL_H__
