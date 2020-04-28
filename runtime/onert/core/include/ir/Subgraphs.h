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

#ifndef __ONERT_IR_SUBGRAPHS_H__
#define __ONERT_IR_SUBGRAPHS_H__

#include <memory>
#include <unordered_map>

#include "ir/Index.h"
#include "util/ObjectManager.h"

namespace onert
{
namespace ir
{

class Graph;

class Subgraphs
{
public:
  Subgraphs() = default;
  Subgraphs(const Subgraphs &obj) = default;
  Subgraphs(Subgraphs &&) = default;
  Subgraphs &operator=(const Subgraphs &) = default;
  Subgraphs &operator=(Subgraphs &&) = default;
  ~Subgraphs() = default;

  /**
   * @brief Put subgraph in the container with a new Index for that
   *
   * @param[in] subg Subgraph to be pushed
   * @param[in] index Index of subgraph to be pushed
   * @return Created
   */
  void push(SubgraphIndex index, const std::shared_ptr<Graph> &subg) { _subgraphs[index] = subg; }

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
   * @return Graph
   */
  const std::shared_ptr<Graph> &at(const SubgraphIndex &index) const
  {
    return _subgraphs.at(index);
  }
  /**
   * @brief Get the subgraph that is associated with the given index
   *
   * @param[in] index Index of the subgraph to be returned
   * @return Graph
   */
  std::shared_ptr<Graph> &at(const SubgraphIndex &index) { return _subgraphs.at(index); }

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
  void iterate(const std::function<void(const SubgraphIndex &, const Graph &)> &fn) const
  {
    for (const auto &e : _subgraphs)
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
  void iterate(const std::function<void(const SubgraphIndex &, Graph &)> &fn)
  {
    for (const auto &e : _subgraphs)
    {
      fn(e.first, *e.second);
    }
  }

  /**
   * @brief Get count of Subgraphs
   *
   * @return count of Subgraphs
   */
  size_t count() { return _subgraphs.size(); }

private:
  std::unordered_map<SubgraphIndex, std::shared_ptr<Graph>> _subgraphs;
};

} // namespace ir
} // namespace onert

#endif // __ONERT_IR_SUBGRAPHS_H__
