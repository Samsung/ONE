/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_IR_MODEL_GRAPH_H__
#define __ONERT_IR_MODEL_GRAPH_H__

#include <memory>
#include <unordered_map>

#include "ir/Index.h"
#include "util/ObjectManager.h"

namespace onert
{
namespace ir
{

class Subgraphs;

class ModelGraph
{
public:
  ModelGraph() = default;
  ModelGraph(const ModelGraph &obj) = default;
  ModelGraph(ModelGraph &&) = default;
  ModelGraph &operator=(const ModelGraph &) = default;
  ModelGraph &operator=(ModelGraph &&) = default;
  ~ModelGraph() = default;

  /**
   * @brief Put Subgraphs in the container with a new Index for that
   *
   * @param[in] subg Subgraph to be pushed
   * @param[in] index Index of subgraph to be pushed
   * @return Created
   */
  void push(ModelIndex index, const std::shared_ptr<Subgraphs> &subgs) { _models[index] = subgs; }

  /**
   * @brief Remove the subgraph that is associated with the given index
   *
   * @param[in] index Index of the subgraph to be removed
   * @return N/A
   */
  void remove(const ModelIndex &index) { _models.erase(index); }

  /**
   * @brief Get the subgraph that is associated with the given index
   *
   * @param[in] index Index of the subgraph to be returned
   * @return Graph
   */
  const std::shared_ptr<Subgraphs> &at(const ModelIndex &index) const { return _models.at(index); }
  /**
   * @brief Get the subgraph that is associated with the given index
   *
   * @param[in] index Index of the subgraph to be returned
   * @return Graph
   */
  std::shared_ptr<Subgraphs> &at(const ModelIndex &index) { return _models.at(index); }

  /**
   * @brief Get the subgraph that is associated with the given index
   *
   * @param[in] index Index of the subgraph to be returned
   * @return true if such entry exists otherwise false
   */
  bool exist(const ModelIndex &index) const
  {
    auto it = _models.find(index);
    return it != _models.end();
  }

  /**
   * @brief Iterate over the container with given function
   *
   * @param[in] fn Function to be run for every container entry
   * @return N/A
   */
  void iterate(const std::function<void(const ModelIndex &, const Subgraphs &)> &fn) const
  {
    for (const auto &e : _models)
    {
      fn(e.first, *e.second);
    }
  }

  /**
   * @brief Iterate over the container with given function
   *
   * @param[in] fn Function to be run for every container entry
   * @return N/A
   */
  void iterate(const std::function<void(const ModelIndex &, Subgraphs &)> &fn)
  {
    for (const auto &e : _models)
    {
      fn(e.first, *e.second);
    }
  }

  /**
   * @brief Get count of Subgraphs
   *
   * @return count of Subgraphs
   */
  size_t count() const { return _models.size(); }

  /**
   * @brief Return the primary subgraph
   *
   * @return std::shared_ptr<Subgraphs> Primary sugraph
   */
  std::shared_ptr<Subgraphs> primary() const { return _models.at(ModelIndex{0}); }

private:
  std::unordered_map<ModelIndex, std::shared_ptr<Subgraphs>> _models;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_MODEL_GRAPH_H__
