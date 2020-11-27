/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _LOCOMOTIV_SESSION_H_
#define _LOCOMOTIV_SESSION_H_

#include "locomotiv/NodeData.h"

#include <loco.h>

#include <memory>
#include <vector>

namespace locomotiv
{

/**
 * @brief Session for loco graph inference
 */
class Session final
{
public:
  Session() = delete;

  /// @brief  Make Session for graph with graph outputs themselves
  Session(loco::Graph *g) : Session(g, loco::output_nodes(g))
  {
    // DO NOTHING
  }

  /**
   * @brief  Make Session for graph with selective custom outputs. Only
   *         subgraph to calculate given outputs would be executed.
   *
   * @note  Set required inputs for given outputs, or inference may fail.
   * @note  custom_outputs don't need to be graph output, but can be any nodes
   *        in the middle of the graph.
   * @warn  This approach may fail in case of graph with control flow
   */
  Session(loco::Graph *g, const std::vector<loco::Node *> &custom_outputs)
    : _graph(g), _outputs(custom_outputs)
  {
    // DO NOTHING
  }

  /// @brief  Make Session by range
  template <typename InputIt>
  Session(loco::Graph *g, InputIt begin, InputIt end) : _graph(g), _outputs(begin, end)
  {
    // DO NOTHING
  }

  /// @brief Free all node annotations of the graph assigned by this Session
  ~Session();

  /// @brief Get number of graph inputs held by this Session
  uint32_t input_size() const { return _graph->inputs()->size(); }

  /**
   * @brief Set graph input at specific index by NodeData.
   *
   * @throw runtime_error In case when another NodeData already annotated for the
   *                      input, and when given data type or shape are not
   *                      congruent with loco node information.
   */
  void set_input(uint32_t index, std::unique_ptr<NodeData> &&data);

  /**
   * @brief Do inference for this session and graph
   *
   * @note Multiple run is possible. Abort program when inputs are not fully set
   *       or invalid calculation found in the middle.
   */
  void infer();

  /// @brief Get number of graph outputs held by this Session
  uint32_t output_size() const { return _outputs.size(); }

  /**
   * @brief Get output of graph as NodeData
   *
   * @note May return nullptr, for example, when graph output not yet calculated
   */
  const NodeData *get_output(uint32_t index);

  const loco::Node *get_output_node(uint32_t index) { return _outputs.at(index); }

private:
  loco::Graph *_graph;
  std::vector<loco::Node *> _outputs;
};

} // namespace locomotiv

#endif // _LOCOMOTIV_SESSION_H_
